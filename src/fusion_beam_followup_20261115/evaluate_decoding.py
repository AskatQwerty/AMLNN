"""Compare greedy and selected beam widths on one immutable full-validation snapshot."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ast
import csv
import gc
import hashlib
import inspect
import json
import math
import sys
import textwrap
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).parent.resolve()
FINAL_SOURCE = BASE.parent / "final_fusion"
sys.path.insert(0,str(FINAL_SOURCE))
from train_run import (PROJECT_DIR,RAW_IMAGE_ROOT,final_code_identity,load_final_grid_features,
                       load_region_prepared_data,load_fusion_region_cache,MetricEvaluator)
from models import FusionCaptionTransformer
from runtime import FusionCaptionDataset,fusion_loader,fusion_device_batch,fusion_seed
from beam_search import fusion_beam_search


def beam_eval_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def beam_eval_file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def beam_eval_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def beam_eval_function_id(functions):
    sources = [ast.dump(ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(function)))))
               for function in functions]
    return beam_eval_hash(sources)


@contextmanager
def beam_evaluation_lock(output_dir):
    path = Path(output_dir)/"evaluation.lock"
    token = uuid.uuid4().hex
    try:
        with path.open("x",encoding="utf-8") as stream:
            json.dump({"token":token,"pid":os.getpid(),"created_utc":datetime.now(timezone.utc).isoformat()},stream)
    except FileExistsError as error:
        raise RuntimeError(f"An evaluation lock exists: {path}. Check its PID; do not remove a live lock.") from error
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("token") == token:
            path.unlink()


def beam_validate_records(records,expected_ids,data,max_steps):
    if [record["image_id"] for record in records] != list(expected_ids):
        raise ValueError("Prediction IDs do not match the exact expected validation order.")
    forbidden = {data["ids"]["pad"],data["ids"]["start"],data["ids"]["end"]}
    vocabulary = data["id_to_token"]
    for record in records:
        tokens = record["token_ids"]
        if (not isinstance(tokens,list) or len(tokens) > max_steps or
                any(type(token) is not int or not 0 <= token < len(vocabulary) or token in forbidden for token in tokens)):
            raise ValueError("A generated caption has invalid IDs, special tokens or length.")
        if record["prediction"] != " ".join(vocabulary[token] for token in tokens):
            raise ValueError("Stored text differs from its generated token IDs.")
        if record["references"] != data["references"]["val"][record["image_id"]]:
            raise ValueError("A raw human reference set changed in a prediction chunk.")


def beam_validate_chunk(chunk,identity_hash,mode,first,last,expected_ids,data,max_steps):
    if (chunk.get("identity_sha256") != identity_hash or chunk.get("mode") != mode or
            chunk.get("first_index") != first or chunk.get("last_index") != last):
        raise ValueError("Prediction chunk belongs to a different snapshot/configuration/range.")
    seconds = chunk.get("generation_seconds")
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Prediction chunk has an invalid timer.")
    beam_validate_records(chunk["records"],expected_ids,data,max_steps)


def beam_load_completed_chunks(folder,identity_hash,mode,dataset,data,config):
    folder = Path(folder)
    chunk_dir = folder/"chunks"
    size = config["batch_size"]
    expected_names = {f"batch_{first:05d}_{min(first+size,len(dataset)):05d}.json"
                      for first in range(0,len(dataset),size)}
    if any(path.name not in expected_names for path in chunk_dir.glob("*.json")):
        raise ValueError("Unexpected prediction chunk filename; use the original evaluation configuration.")
    records,seconds,missing = [],0.0,False
    for first in range(0,len(dataset),size):
        last = min(first+size,len(dataset))
        path = chunk_dir/f"batch_{first:05d}_{last:05d}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise ValueError("A prediction chunk follows a missing earlier chunk; refusing to hide a gap.")
        chunk = json.loads(path.read_text(encoding="utf-8"))
        beam_validate_chunk(chunk,identity_hash,mode,first,last,dataset.image_ids[first:last],data,config["max_steps"])
        records.extend(chunk["records"])
        seconds += chunk["generation_seconds"]
    return records,seconds


@torch.no_grad()
def beam_evaluate_mode(model,dataset,data,config,device,evaluator,mode,output_dir,identity_hash,
                       max_new_batches=None,_after_chunk=None):
    """Resume atomic chunks; partial generation never produces a corpus metric."""
    beam_sizes = config.get("beam_sizes",[3])
    if dataset.split != "val" or mode not in ["greedy",*[f"beam{size}" for size in beam_sizes]]:
        raise ValueError("Use greedy or a configured beam width on the labeled validation split.")
    if max_new_batches is not None and (type(max_new_batches) is not int or max_new_batches < 1):
        raise ValueError("max_new_batches must be a positive optional work limit.")
    folder = Path(output_dir)/mode
    (folder/"chunks").mkdir(parents=True,exist_ok=True)
    records,generation_seconds = beam_load_completed_chunks(folder,identity_hash,mode,dataset,data,config)
    device = torch.device(device)
    use_amp = bool(config["amp"] and device.type == "cuda")
    model.eval()
    fusion_seed(config["seed"])
    loader = iter(fusion_loader(dataset,config["batch_size"],indices=list(range(len(records),len(dataset))),
                               seed=config["seed"],pin_memory=use_amp))
    new_batches = 0
    try:
        while len(records) < len(dataset):
            if max_new_batches is not None and new_batches >= max_new_batches:
                break
            if device.type == "cuda": torch.cuda.synchronize(device)
            started = time.perf_counter()
            batch = next(loader)
            moved = fusion_device_batch(batch,device)
            extras = {"region_mask":moved["region_mask"],"region_boxes":moved["region_boxes"],
                      "global_features":moved["global_features"]}
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                if mode == "greedy":
                    tokens = model.greedy(moved["grid"],moved["regions"],data["ids"]["start"],data["ids"]["end"],
                                          max_steps=config["max_steps"],**extras)
                else:
                    tokens = fusion_beam_search(model,moved["grid"],moved["regions"],data["ids"]["start"],data["ids"]["end"],
                        max_steps=config["max_steps"],beam_size=int(mode[4:]),length_penalty=config["length_penalty"],**extras)
            if len(tokens) != len(batch["image_ids"]):
                raise ValueError("Decoder returned a different number of captions than images.")
            chunk_records = [{"image_id":int(image_id),"prediction":" ".join(data["id_to_token"][token] for token in row),
                              "token_ids":row,"references":references}
                             for image_id,row,references in zip(batch["image_ids"],tokens,batch["references"])]
            if device.type == "cuda": torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            first,last = len(records),len(records)+len(chunk_records)
            chunk = {"identity_sha256":identity_hash,"mode":mode,"first_index":first,"last_index":last,
                     "generation_seconds":seconds,"records":chunk_records}
            beam_validate_chunk(chunk,identity_hash,mode,first,last,dataset.image_ids[first:last],data,config["max_steps"])
            beam_eval_json(folder/"chunks"/f"batch_{first:05d}_{last:05d}.json",chunk)
            records.extend(chunk_records)
            generation_seconds += seconds
            new_batches += 1
            beam_eval_json(folder/"progress.json",{"status":"generating","identity_sha256":identity_hash,
                "completed_images":len(records),"required_images":len(dataset),"generation_seconds":generation_seconds,
                "updated_utc":datetime.now(timezone.utc).isoformat()})
            if new_batches == 1 or new_batches % 20 == 0 or len(records) == len(dataset):
                speed = len(records)/max(generation_seconds,1e-9)
                print(f"{mode}: {len(records):,}/{len(dataset):,} | {speed:.1f} img/s | generation {generation_seconds:.1f} s",flush=True)
            if _after_chunk is not None: _after_chunk(chunk)
            del batch,moved,tokens
        if len(records) != len(dataset):
            partial = {"status":"partial","identity_sha256":identity_hash,"mode":mode,"completed_images":len(records),
                       "required_images":len(dataset),"generation_seconds":generation_seconds,"metrics_computed":False}
            beam_eval_json(folder/"progress.json",partial)
            return partial
        beam_validate_records(records,dataset.image_ids,data,config["max_steps"])
        predictions_hash = beam_eval_hash(records)
        predictions_path = folder/"validation_predictions.json"
        if predictions_path.exists():
            if beam_eval_hash(json.loads(predictions_path.read_text(encoding="utf-8"))) != predictions_hash:
                raise ValueError("Final prediction file differs from its validated chunks.")
        else:
            beam_eval_json(predictions_path,records)
        metrics_path = folder/"metrics.json"
        if metrics_path.exists():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (result.get("identity_sha256") != identity_hash or result.get("predictions_sha256") != predictions_hash or
                    result.get("status") != "complete" or result.get("mode") != mode or result.get("images") != len(dataset) or
                    set(result.get("scores",{})) != {"bleu1","bleu4","cider_d"} or
                    not all(math.isfinite(float(value)) for value in result["scores"].values())):
                raise ValueError("Saved metrics do not match the exact current full prediction corpus.")
            return result
        started = time.perf_counter()
        scores = evaluator.score(records,expected_ids=dataset.image_ids)
        metric_seconds = time.perf_counter()-started
        if set(scores) != {"bleu1","bleu4","cider_d"} or not all(math.isfinite(float(value)) for value in scores.values()):
            raise ValueError("The verified scorer returned unexpected or nonfinite corpus scores.")
        result = {"status":"complete","identity_sha256":identity_hash,"mode":mode,"images":len(records),
                  "scores":scores,"generation_seconds":generation_seconds,"metric_seconds":metric_seconds,
                  "evaluation_seconds":generation_seconds+metric_seconds,"predictions_sha256":predictions_hash,
                  "average_words":sum(len(record["token_ids"]) for record in records)/len(records),
                  "predictions_file":str((folder/"validation_predictions.json").resolve()),"metrics_computed":True}
        beam_eval_json(metrics_path,result)
        beam_eval_json(folder/"progress.json",result)
        return result
    except BaseException as error:
        beam_eval_json(folder/"progress.json",{"status":"interrupted","identity_sha256":identity_hash,
            "completed_images":len(records),"required_images":len(dataset),"generation_seconds":generation_seconds,
            "error_type":type(error).__name__,"error":str(error),"metrics_computed":False})
        raise


def load_beam_snapshot(snapshot_dir,data,evaluator,dataset,code_identity):
    """Use only the copied full checkpoint; never combine live weights and live metadata."""
    snapshot_dir = Path(snapshot_dir)
    metadata = json.loads((snapshot_dir/"snapshot.json").read_text(encoding="utf-8"))
    path = snapshot_dir/"best.pt"
    expected_hash = metadata["full_checkpoint_sha256"]
    if beam_eval_file_sha(path) != expected_hash:
        raise ValueError("The pinned full checkpoint SHA256 changed.")
    payload = torch.load(path,map_location="cpu",weights_only=False)
    if beam_eval_file_sha(path) != expected_hash:
        raise ValueError("The checkpoint changed while it was being read.")
    identity,state = payload["identity"],payload["state"]
    assert payload["format_version"] == 1
    assert identity["data"] == data["provenance"] and identity["metrics"] == evaluator.metadata
    assert identity["validation_ids"] == data["split_ids"]["val"] == dataset.image_ids
    assert len(dataset) == 5000 and len(set(dataset.image_ids)) == 5000
    assert identity["train_ids"] == sorted(data["split_ids"]["train"]) and len(identity["train_ids"]) == 113287
    assert not set(dataset.image_ids)&(set(data["split_ids"]["train"])|set(data["split_ids"]["test"]))
    assert identity["grid_and_region_data"]["val"] == dataset.provenance
    trained = identity["config"]
    assert trained["code"]["models_sha256"] == code_identity["models_sha256"]
    assert trained["code"]["region_producer_code_id"] == code_identity["region_producer_code_id"]
    assert metadata["snapshot_epoch"] == state["epoch"] and metadata["best_epoch"] == state["best_epoch"]
    assert state["epoch"] == state["best_epoch"] and metadata["best_record"] == state["best_record"]
    if not all(bool(torch.isfinite(value).all()) for value in payload["model"].values() if value.is_floating_point()):
        raise FloatingPointError("Pinned checkpoint has nonfinite model weights.")
    return payload,metadata


def run_fusion_decoding_comparison(project_dir,snapshot_dir,output_dir,raw_root=None,code_identity=None,
                                   batch_size=16,amp=True,length_penalty=0.6,device=None,max_new_batches=None,
                                   beam_sizes=(3,5)):
    """Read-only evaluation. Defaults compare greedy, beam3 and beam5 on all 5,000 validation IDs."""
    if type(batch_size) is not int or batch_size < 1 or not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError("Use a positive batch size and finite nonnegative length penalty.")
    if (not isinstance(beam_sizes,(list,tuple)) or not beam_sizes or
            any(type(size) is not int or size < 2 for size in beam_sizes) or len(beam_sizes) != len(set(beam_sizes))):
        raise ValueError("beam_sizes must contain distinct integer widths >=2, for example (3,5).")
    beam_sizes = list(beam_sizes)
    if code_identity is None:
        code_identity = globals().get("EXPECTED_CODE_IDENTITY")
    if not code_identity:
        raise ValueError("Pass the verified model/cache code identity from this notebook or runner.")
    project_dir,snapshot_dir,output_dir = Path(project_dir),Path(snapshot_dir),Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not (output_dir/"evaluation_identity.json").is_file():
        raise ValueError("Use a dedicated empty evaluation directory, or resume one with its saved identity.")
    output_dir.mkdir(parents=True,exist_ok=True)
    dataset = model = None
    with beam_evaluation_lock(output_dir):
        try:
            data = load_region_prepared_data(project_dir,raw_root)
            evaluator = MetricEvaluator(data["text_settings"])
            grids = load_final_grid_features(data,project_dir)
            regions = load_fusion_region_cache(data,project_dir/"cache/features/fasterrcnn_fusion_expanded_v1",
                                              producer_code_id=code_identity["region_producer_code_id"])
            dataset = FusionCaptionDataset(data,grids,regions,"val",seed=42)
            payload,snapshot = load_beam_snapshot(snapshot_dir,data,evaluator,dataset,code_identity)
            device = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
            details = {"device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cpu_threads":torch.get_num_threads()}
            config = {"seed":42,"batch_size":batch_size,"amp":bool(amp),"amp_enabled":bool(amp and device.type == "cuda"),
                      "max_steps":33,"beam_sizes":beam_sizes,"length_penalty":length_penalty,
                      "length_penalty_formula":"((5+L)/6)^alpha; L excludes START and includes EOS when emitted",
                      "methods":["greedy",*[f"beam{size}" for size in beam_sizes]],
                      "selection":"Fixed beam-width comparison on validation; no model training and no test evaluation"}
            identity = {"snapshot_full_checkpoint_sha256":snapshot["full_checkpoint_sha256"],
                        "snapshot_epoch":snapshot["snapshot_epoch"],"model_source_sha256":code_identity["models_sha256"],
                        "data":data["provenance"],"validation_ids":dataset.image_ids,"cache":dataset.provenance,
                        "scorer":evaluator.metadata,"config":config,"device":details,
                        "beam_implementation_id":beam_eval_function_id([fusion_beam_search]),
                        "evaluation_implementation_id":beam_eval_function_id([beam_validate_records,beam_validate_chunk,
                            beam_load_completed_chunks,beam_evaluate_mode])}
            identity_path = output_dir/"evaluation_identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Existing evaluation belongs to another snapshot/settings/device/code; use a new directory.")
            beam_eval_json(identity_path,identity)
            beam_eval_json(output_dir/"snapshot_metadata.json",snapshot)
            beam_eval_json(output_dir/"scorer.json",evaluator.metadata)
            beam_eval_json(output_dir/"validation_image_ids.json",dataset.image_ids)
            trained = payload["identity"]["config"]
            model = FusionCaptionTransformer(len(data["id_to_token"]),data["ids"]["pad"],
                grid_feature_dim=trained["grid_feature_dim"],region_feature_dim=trained["region_feature_dim"],
                d_model=trained["d_model"],num_layers=trained["num_layers"],nhead=trained["nhead"],
                dim_feedforward=trained["dim_feedforward"],dropout=trained["dropout"],
                grid_size=trained["grid_size"],max_regions=trained["max_regions"],max_positions=34)
            model.load_state_dict(payload["model"],strict=True)
            del payload
            model = model.to(device).eval()
            model.requires_grad_(False)
            identity_hash = beam_eval_hash(identity)
            results = {mode:beam_evaluate_mode(model,dataset,data,config,device,evaluator,mode,output_dir,
                                              identity_hash,max_new_batches=max_new_batches)
                       for mode in config["methods"]}
            complete = all(result["status"] == "complete" for result in results.values())
            comparison = {"status":"complete" if complete else "partial","identity_sha256":identity_hash,
                "snapshot_epoch":snapshot["snapshot_epoch"],"snapshot_full_checkpoint_sha256":snapshot["full_checkpoint_sha256"],
                "snapshot_scope":snapshot.get("scope","Pinned best at snapshot time; ongoing training can produce a later best."),
                "validation_images":5000,"device":details,"results":results,"test_evaluated":False,
                "model_training_performed":False,"original_checkpoints_modified":False,
                "timing_note":"Per-mode generation includes cache loading, device copies and decoding; scoring is separate. Model loading, artifact I/O and pauses are excluded. Main training continues on the same GPU, so these times are not an isolated architecture benchmark."}
            if complete:
                comparison["differences_from_greedy"] = {
                    mode:{key:result["scores"][key]-results["greedy"]["scores"][key]
                          for key in ["cider_d","bleu1","bleu4"]}
                    for mode,result in results.items() if mode != "greedy"}
                comparison["beam_deltas"] = comparison["differences_from_greedy"]
                comparison["selected_mode_by_validation_cider"] = max(
                    results,key=lambda mode:results[mode]["scores"]["cider_d"])
                # Keep the original beam3 field readable by the existing comparison display.
                if "beam3" in results:
                    comparison["beam_minus_greedy"] = comparison["differences_from_greedy"]["beam3"]
                temporary = output_dir/"comparison.csv.tmp"
                with temporary.open("w",newline="",encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream,fieldnames=["mode","snapshot_epoch","images","bleu1","bleu4","cider_d",
                        "generation_seconds","metric_seconds","evaluation_seconds","average_words","device"])
                    writer.writeheader()
                    for mode,result in results.items():
                        writer.writerow({"mode":mode,"snapshot_epoch":snapshot["snapshot_epoch"],"images":result["images"],
                            **result["scores"],**{key:result[key] for key in ["generation_seconds","metric_seconds","evaluation_seconds","average_words"]},
                            "device":details["gpu"] or details["device"]})
                temporary.replace(output_dir/"comparison.csv")
            beam_eval_json(output_dir/"comparison.json",comparison)
            return comparison
        finally:
            if dataset is not None: dataset.close()
            model = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--project",type=Path,default=PROJECT_DIR)
    parser.add_argument("--raw",type=Path,default=RAW_IMAGE_ROOT)
    parser.add_argument("--snapshot",type=Path,default=BASE/"best_snapshot")
    parser.add_argument("--output",type=Path,default=BASE/"validation_comparison")
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--beam-sizes",type=int,nargs="+",default=[3,5])
    parser.add_argument("--max-new-batches",type=int)
    args = parser.parse_args()
    if not args.execute:
        print("Evaluation remains off. Add --execute to compare greedy and the selected beam widths on the pinned snapshot.")
        return
    torch.set_num_threads(2)
    result = run_fusion_decoding_comparison(args.project,args.snapshot,args.output,raw_root=args.raw,
        code_identity=final_code_identity(source_dir=FINAL_SOURCE),batch_size=args.batch_size,
        max_new_batches=args.max_new_batches,beam_sizes=args.beam_sizes)
    print(json.dumps({"status":result["status"],"snapshot_epoch":result["snapshot_epoch"],
                      "beam_deltas":result.get("beam_deltas"),
                      "selected_mode_by_validation_cider":result.get("selected_mode_by_validation_cider")},indent=2))


if __name__ == "__main__":
    main()
