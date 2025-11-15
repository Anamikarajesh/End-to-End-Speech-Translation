import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ffmpeg_path = Path('/home/user/anamika_ml_project/tools/ffmpeg-7.0.2-amd64-static/ffmpeg')
dataset_dir = Path('/home/user/anamika_ml_project/dataset/Marathi')

if not ffmpeg_path.is_file():
    raise SystemExit(f'Missing ffmpeg binary at {ffmpeg_path}')
if not dataset_dir.is_dir():
    raise SystemExit(f'Dataset directory missing: {dataset_dir}')

files = sorted(dataset_dir.rglob('*.3gp'))
total = len(files)
print(f'Found {total} .3gp files to process')

converted = 0
skipped = 0
errors = []

log_every = 200


def convert(one_path: Path):
    wav_path = one_path.with_suffix('.wav')
    if wav_path.exists():
        return 'skipped', str(one_path)
    cmd = [
        str(ffmpeg_path),
        '-y',
        '-loglevel', 'error',
        '-i', str(one_path),
        '-ac', '1',
        '-ar', '16000',
        str(wav_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return 'error', str(one_path), result.stderr.decode('utf-8', 'ignore')
    return 'converted', str(one_path)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(convert, path): path for path in files}
    for idx, future in enumerate(as_completed(futures), 1):
        status = future.result()
        if status[0] == 'converted':
            converted += 1
        elif status[0] == 'skipped':
            skipped += 1
        else:
            errors.append((status[1], status[2]))
        if idx % log_every == 0 or idx == total:
            print(f'Processed {idx}/{total} files (converted={converted}, skipped={skipped}, errors={len(errors)})')

print('Conversion complete')
print(f'Converted: {converted}, skipped: {skipped}, errors: {len(errors)}')

if errors:
    print('Sample error:')
    for path, msg in errors[:5]:
        print(path)
        print(msg[:500])