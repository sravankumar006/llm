"""Package the CommandLLM repository into commandllm_colab.zip for Google Colab."""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = ROOT / "commandllm_colab.zip"

INCLUDE_DIRS = ["core", "data", "scripts", "server"]
INCLUDE_FILES = [
    "colab_requirements.txt",
    "requirements.txt",
    "README.md",
    "colab_train.ipynb",
]

print(f"Creating {ZIP_NAME}...")

count = 0
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename in INCLUDE_FILES:
        filepath = ROOT / filename
        if filepath.exists():
            zf.write(filepath, arcname=filename)
            count += 1
            print(f" [+] {filename}")

    for dir_name in INCLUDE_DIRS:
        dir_path = ROOT / dir_name
        for root, dirs, files in os.walk(dir_path):
            if "__pycache__" in root or ".pytest_cache" in root:
                continue
            for f in files:
                file_path = Path(root) / f
                rel_path = file_path.relative_to(ROOT)
                zf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
                count += 1

size_mb = ZIP_NAME.stat().st_size / (1024 * 1024)
print(f"Successfully packaged {count} files into {ZIP_NAME.name} ({size_mb:.2f} MB)")
