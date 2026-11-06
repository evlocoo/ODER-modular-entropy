from pathlib import Path
import hashlib, json, os
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'inputs' / 'receiver_extensions'
def verify_inputs():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, entry in manifest.items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert actual == entry['sha256'], f'Input hash mismatch: {name}'
    return manifest
def save_result(name, result):
    folder = ROOT / 'outputs' / 'runs' / os.environ.get('ODER_RUN_NAME', 'interactive_extensions') / 'extensions'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(result, indent=2, default=lambda x: x.item()) + '\n')
    return path.relative_to(ROOT).as_posix()
