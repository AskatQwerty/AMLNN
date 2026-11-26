"""Recompute the published follow-up metrics on CPU from saved captions."""
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/stage5_handoff"))
from metrics import MetricEvaluator

# Use the original tokenizer and all human references, without vocabulary mapping.
settings=json.loads((ROOT/"metadata/preparation/text_report.json").read_text(encoding="utf-8"))
evaluator=MetricEvaluator(settings["config"]["settings"])
results=ROOT/"results/followup_20261126"

# Each score is one corpus-level calculation over the complete saved prediction file.
for subset in ["resolution/full_validation","resolution/pilot_100","finetuning/validation_100"]:
    folder=results/subset
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    rows=list(csv.DictReader((folder/"metrics.csv").open(encoding="utf-8",newline="")))
    print("\n"+subset)
    for row in rows:
        name=row["mode"] if "mode" in row else row["model"]
        records=json.loads((folder/f"{name}_predictions.json").read_text(encoding="utf-8"))
        scores=evaluator.score(records,expected_ids=manifest["image_ids"])
        columns={"bleu1":"BLEU-1","bleu4":"BLEU-4","cider_d":"CIDEr-D"}
        for metric,value in scores.items():
            expected=float(row[metric] if metric in row else row[columns[metric]])
            assert abs(value-expected)<1e-10,(subset,name,metric,value,expected)
        print(f"{name}: n={len(records)}, BLEU-1={scores['bleu1']:.5f}, "
              f"BLEU-4={scores['bleu4']:.5f}, CIDEr-D={scores['cider_d']:.5f}")

print("\nAll saved corpus metrics reproduced. No models, training or test images used.")
