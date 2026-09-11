"""Copy only console source into this repository; never settings, media or caches."""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
destination = Path(__file__).resolve().parents[1] / 'console'
files = ['app.py', 'requirements.txt', 'README.md', 'sam3_local_runner.py',
         'sam3_runner_template.py', 'tts_sapi.ps1', 'run_windows.bat',
         'run_local_network.bat', 'Run_RunPod_Media_Console_Web.bat',
         'static/app.js', 'static/index.html', 'static/styles.css',
         'tests/test_security_and_h3.py', 'tests/test_samimate_h3.py',
         'tests/test_samimate_ui.cjs', 'tests/test_h3_frames_ui.cjs',
         'tests/test_h3_turbo_ui.cjs']
for name in files:
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source / name, target)
print(f'Synced {len(files)} console source files')
