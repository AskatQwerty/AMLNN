"""Generate a caption from the saved final model and one original photograph."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from models import FusionCaptionTransformer
from image_features import build_frozen_fusion_encoders,extract_fusion_image_features
from beam_search import fusion_beam_search


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def load_final_model(model_dir,device):
    model_dir=Path(model_dir)
    inference=json.loads((model_dir/'inference_config.json').read_text(encoding='utf-8'))
    vocabulary=json.loads((model_dir/'vocabulary.json').read_text(encoding='utf-8'))
    source=json.loads((model_dir/'source_run_config.json').read_text(encoding='utf-8'))
    config=source['config']

    # Historical snapshot paths in the metadata are provenance, not dependencies.
    weights_path=model_dir/'weights.pt'
    actual_sha=file_sha256(weights_path)
    if actual_sha!=inference['weights_sha256']:
        raise ValueError('The final model weights do not match inference_config.json.')
    if file_sha256(model_dir/'vocabulary.json')!=inference['vocabulary_sha256']:
        raise ValueError('The saved vocabulary differs from the final model vocabulary.')
    if file_sha256(model_dir/'source_run_config.json')!=inference['source_run_config_sha256']:
        raise ValueError('The model architecture configuration differs from the saved run.')

    token_ids={name:vocabulary['token_to_id'][f'<{name}>'] for name in ['pad','unk','start','end']}
    model=FusionCaptionTransformer(
        vocab_size=len(vocabulary['id_to_token']),pad_id=token_ids['pad'],
        grid_feature_dim=config['grid_feature_dim'],region_feature_dim=config['region_feature_dim'],
        d_model=config['d_model'],num_layers=config['num_layers'],nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],dropout=config['dropout'],
        grid_size=config['grid_size'],max_regions=config['max_regions'],max_positions=34,
    )
    state=torch.load(weights_path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model=model.to(device).requires_grad_(False).eval()
    return model,vocabulary,token_ids,inference,actual_sha


@torch.inference_mode()
def caption_image(image_path,model_dir,device='auto',allow_download=False,torch_cache=None):
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'cpu'
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or --device auto.')
    if torch_cache is not None:
        torch.hub.set_dir(str(Path(torch_cache).resolve()))

    torch.manual_seed(42)
    torch.set_num_threads(2)
    started=time.perf_counter()
    model,vocabulary,token_ids,inference,weights_sha=load_final_model(model_dir,device)
    encoders=build_frozen_fusion_encoders(device,project_dir=model_dir,allow_download=allow_download)
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    loading_seconds=time.perf_counter()-started

    started=time.perf_counter()
    features,audit=extract_fusion_image_features(image_path,encoders)
    # This is the final validation-selected decoding configuration.
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        tokens=fusion_beam_search(
            model,features['grid'],features['regions'],token_ids['start'],token_ids['end'],
            max_steps=33,region_mask=features['region_mask'],region_boxes=features['region_boxes'],
            global_features=features['global_features'],beam_size=5,length_penalty=0.0,
        )[0]
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    seconds=time.perf_counter()-started
    forbidden={token_ids['pad'],token_ids['start'],token_ids['end']}
    assert not any(token in forbidden for token in tokens)
    return {
        'caption':' '.join(vocabulary['id_to_token'][token] for token in tokens),
        'token_ids':tokens,'image':str(Path(image_path).resolve()),
        'model_epoch':inference['snapshot_epoch'],'weights_sha256':weights_sha,
        'generation':{'method':'beam','beam_size':5,'length_penalty':0.0,'max_steps':33},
        'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'torch':torch.__version__,'loading_seconds':loading_seconds,'inference_seconds':seconds,
        'valid_regions':audit['valid_regions'],'region_fallback':audit['region_fallback'],
        'image_sha256':audit['image_sha256'],
        'uses_precomputed_feature_arrays':False,'training_performed':False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True,help='Original image file.')
    parser.add_argument('--model-dir',type=Path,required=True,help='Folder containing weights.pt and the three final-model JSON files.')
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda or cuda:0.')
    parser.add_argument('--torch-cache',type=Path,help='Optional torch hub root containing checkpoints/ with the two encoder .pth files.')
    parser.add_argument('--allow-download',action='store_true',help='Allow torchvision to download missing pretrained CNN weights.')
    parser.add_argument('--output',type=Path,help='Optional JSON result file. No feature cache is written.')
    args=parser.parse_args()
    result=caption_image(args.image,args.model_dir,args.device,args.allow_download,args.torch_cache)
    encoded=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded+'\n',encoding='utf-8')
    print(encoded)


if __name__=='__main__':
    main()
