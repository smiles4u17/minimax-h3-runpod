"""External SAM3 runner template for RunPod Media Console Web.

Command example:
python sam3_runner_template.py --input {input_json} --output {output_json}

Replace the center-ellipse fallback with your real SAM3 import/inference.
"""
import argparse, json, time, uuid
from pathlib import Path
from PIL import Image, ImageDraw
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
job=json.loads(Path(a.input).read_text(encoding='utf-8')); frame=Path(job['frame_path']); out_dir=Path(job.get('output_dir') or frame.parent); out_dir.mkdir(parents=True,exist_ok=True)
img=Image.open(frame).convert('RGB'); w,h=img.size; mask=Image.new('L',img.size,0); d=ImageDraw.Draw(mask); d.ellipse([int(w*.25),int(h*.18),int(w*.75),int(h*.86)],fill=255)
stamp=f"external_sam3_{int(time.time())}_{uuid.uuid4().hex[:8]}"
mask_path=out_dir/f"{stamp}_mask.png"; overlay_path=out_dir/f"{stamp}_overlay.png"; cutout_path=out_dir/f"{stamp}_cutout.png"; masked_path=out_dir/f"{stamp}_masked.png"
mask.save(mask_path); red=Image.new('RGBA',img.size,(255,0,0,110)); Image.composite(red,img.convert('RGBA'),mask).save(overlay_path); cutout=img.convert('RGBA'); cutout.putalpha(mask); cutout.save(cutout_path); Image.composite(img, Image.new('RGB',img.size,(0,0,0)), mask).save(masked_path)
Path(a.output).write_text(json.dumps({'mask_path':str(mask_path),'overlay_path':str(overlay_path),'cutout_path':str(cutout_path),'masked_image_path':str(masked_path)},indent=2),encoding='utf-8')
