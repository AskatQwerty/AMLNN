"""Verify the pristine transfer before using its prepared pickle cache."""
import hashlib
import json
import sys
from pathlib import Path


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("A manifest path escapes the bundle directory.")
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing or changed file: {item['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {item['path']}")
    print(f"Verified {len(manifest['files'])} delivered files. Keep the pristine starter unchanged.")
    return True


if __name__ == "__main__":
    verify_bundle(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
