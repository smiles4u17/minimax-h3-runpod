import hashlib,json,os,signal,time,urllib.request,uuid
from pathlib import Path
signal.alarm(1800)
root=Path('/runpod-volume')
if not os.path.ismount(root): raise RuntimeError('Network volume is not mounted')
name='qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
expected_size=27141342152
expected_sha='bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923'
revision='e5eb578a89295337b8ff433a035929ce0279e0b6'
url='https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/'+revision+'/text_encoders/'+name
target=root/'models'/'text_encoders'/name
status=root/'diagnostics'/'universal_encoder_install.json'
started=time.monotonic()
def report(stage,**kw):
 d=dict(stage=stage,target=str(target),elapsed_seconds=round(time.monotonic()-started),expected_size=expected_size,expected_sha256=expected_sha,source_revision=revision,**kw)
 status.parent.mkdir(parents=True,exist_ok=True)
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(d));os.replace(tmp,status)
 print(json.dumps(d),flush=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if target.exists():
 if target.stat().st_size!=expected_size or digest(target)!=expected_sha:raise RuntimeError('Existing target is different; refusing overwrite')
 report('COMPLETED',sha256=expected_sha,already_present=True)
else:
 target.parent.mkdir(parents=True,exist_ok=True)
 v=os.statvfs(root)
 if v.f_bavail*v.f_frsize < expected_size+1024**3:raise RuntimeError('Insufficient free volume space')
 part=target.with_name(name+'.part-'+uuid.uuid4().hex)
 report('DOWNLOADING',bytes=0)
 h=hashlib.sha256();total=0;next_report=256*1024*1024
 try:
  with urllib.request.urlopen(url,timeout=60) as response,part.open('xb') as f:
   for b in iter(lambda:response.read(8*1024*1024),b''):
    total+=len(b)
    if total>expected_size:raise RuntimeError('Unexpected source size')
    h.update(b);f.write(b)
    if total>=next_report:report('DOWNLOADING',bytes=total);next_report=total+256*1024*1024
   f.flush();os.fsync(f.fileno())
  if total!=expected_size or h.hexdigest()!=expected_sha:raise RuntimeError('Source size or SHA256 mismatch')
  report('VERIFYING_STORED_BYTES',bytes=total)
  stored=digest(part)
  if stored!=expected_sha:raise RuntimeError('Stored-byte SHA256 mismatch')
  os.replace(part,target)
  report('COMPLETED',bytes=total,sha256=stored)
 except Exception as e:
  report('FAILED',bytes=total,error=str(e))
  raise

