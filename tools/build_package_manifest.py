"""Record hashes for the tracked and visible untracked archive files."""
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path

root=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode('utf-8').split('\0')
files=[]
for name in sorted(set(names)):
    if not name or name=='package_manifest.json':
        continue
    path=root/name
    if not path.is_file():
        continue
    files.append({'path':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
manifest={
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'purpose':'Lightweight final project Git archive',
    'excludes':['package_manifest.json','.git/','Git-ignored datasets, numeric features, weights and archives'],
    'file_count':len(files),
    'bytes':sum(item['bytes'] for item in files),
    'files':files,
}
(root/'package_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'file_count':manifest['file_count'],'bytes':manifest['bytes'],'manifest':'package_manifest.json'}))
