"""Download only the public M1 held-in files needed by TCN.cross_day_tcn."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/000941'))
    args = parser.parse_args()
    url = 'https://api.dandiarchive.org/api/dandisets/000941/versions/draft/assets/?page_size=100'
    assets = []
    while url:
        with urlopen(url, timeout=60) as response:
            page = json.load(response)
        assets.extend(a for a in page['results'] if a['path'].endswith('.nwb') and
                      any(group in a['path'] for group in ('held-in-calib/', 'held-in-minival/')))
        url = page.get('next')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'download_manifest.json').write_text(json.dumps(assets, indent=2))
    for asset in assets:
        destination = args.out / asset['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == asset['size']:
            print('Already complete:', destination.name, flush=True)
            continue
        temporary = destination.with_suffix('.nwb.part')
        with urlopen(f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/", timeout=120) as source:
            with temporary.open('wb') as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
        if temporary.stat().st_size != asset['size']:
            raise RuntimeError(f'Incomplete download: {temporary}')
        temporary.replace(destination)
        print('Downloaded:', destination.name, flush=True)
    print(f'Verified {len(assets)} files by byte size.', flush=True)


if __name__ == '__main__':
    main()
