"""Run the preserved local V8 pipeline; this is not an exact final-submission replay."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def checkpoint(name):
    import torch
    manifest = json.loads((ROOT/'models/manifest.json').read_text())
    entry = manifest[name]
    parts = []
    for part in entry['parts']:
        data = (ROOT/'models'/part['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != part['sha256']:
            raise ValueError(f'Checkpoint part checksum mismatch: {part["file"]}')
        parts.append(data)
    data = b''.join(parts)
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Combined checkpoint checksum mismatch')
    return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)


def load_models(device):
    import torch
    from torch import nn
    from src.aponeurosis_model_v2 import AponeurosisUNetV2
    from src.fascicle_orientation_unet import FascicleOrientationUNet
    apo = AponeurosisUNetV2(in_channels=1,base_channels=32)
    fascicle = FascicleOrientationUNet(in_channels=1,out_channels=3,base_channels=32)
    apo.load_state_dict(checkpoint('aponeurosis')['state_dict'])
    fascicle.load_state_dict(checkpoint('orientation')['state_dict'])
    heads = []
    for name in ['selection_support','measurement_support']:
        head = nn.Sequential(nn.Conv2d(fascicle.head.in_channels,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))
        head.load_state_dict(checkpoint(name)['state_dict'])
        heads.append(head.to(device).eval())
    return apo.to(device).eval(),fascicle.to(device).eval(),*heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v8')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--check-models',action='store_true',help='Load all model states and pipeline imports without image inference')
    parser.add_argument('--depth-mm',type=float,help='Optional manually verified full ruler depth')
    args = parser.parse_args()
    if not args.check_models and args.image is None:
        parser.error('--image is required unless --check-models is used')
    from src.test_set_inference_pipeline_v8 import run_single_image
    apo,fascicle,selection,measurement = load_models(args.device)
    if args.check_models:
        print('All four model states and the V8 dependency chain loaded; no inference performed.')
        return
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    def read_text(image):
        result,_ = engine(image)
        return ' '.join(str(item[1]) for item in (result or []))
    result = run_single_image(args.image,apo,fascicle,args.device,read_text=read_text,
                              support_head=selection,measurement_support_head=measurement,
                              depth_mm=args.depth_mm,weighted_fl=True,kappa_max=100,
                              fl_extension_slope_fraction=.75,dense_seeds=True)
    summary = result['measurements'].get('summary',{})
    calibration = result['calibration_v3']
    record = {'image_id':args.image.name,'workflow':'preserved local V8; final notebook-only patches not applied',
              'measurement_status':result['measurements']['status'],
              'pair':None if result['region_selection']['selected_region'] is None else result['region_selection']['selected_region']['pair_name'],
              'measurements':summary,'calibration':{k:calibration.get(k) for k in
                  ['status','source','reason','depth_mm','zero_y','last_y','mm_per_raw_px','ocr_reads']}}
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [clean(v) for v in value]
        if isinstance(value,np.ndarray): return clean(value.tolist())
        if isinstance(value,np.generic): return clean(value.item())
        if isinstance(value,float) and not np.isfinite(value): return None
        return value
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/(args.image.stem+'.json')).write_text(json.dumps(clean(record),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(clean(record),indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
