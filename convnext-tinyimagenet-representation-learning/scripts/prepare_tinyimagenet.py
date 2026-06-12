import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path


URLS = [
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract Tiny ImageNet-200")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--force", action="store_true", help="Download even if the zip already exists")
    return parser.parse_args()


def reporthook(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    sys.stdout.write(f"\rDownloading: {percent:5.1f}%")
    sys.stdout.flush()


def download(url: str, output_zip: Path) -> bool:
    try:
        print(f"Trying: {url}")
        urllib.request.urlretrieve(url, output_zip, reporthook=reporthook)
        print("\nDownload complete.")
        return True
    except Exception as exc:
        print(f"\nFailed: {exc}")
        if output_zip.exists():
            output_zip.unlink()
        return False


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = data_dir / "tiny-imagenet-200"
    if (extracted_dir / "wnids.txt").exists() and not args.force:
        print(f"Tiny ImageNet already exists at: {extracted_dir}")
        return

    output_zip = data_dir / "tiny-imagenet-200.zip"
    if not output_zip.exists() or args.force:
        success = False
        for url in URLS:
            success = download(url, output_zip)
            if success:
                break
        if not success:
            raise RuntimeError("Could not download Tiny ImageNet from any configured URL.")
    else:
        print(f"Using existing zip: {output_zip}")

    print(f"Extracting to: {data_dir}")
    with zipfile.ZipFile(output_zip, "r") as zf:
        zf.extractall(data_dir)

    print(f"Done. Dataset root: {extracted_dir}")


if __name__ == "__main__":
    main()
