"""Release smoke tests for the immutable shared ML runtime."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path


IMPORTS = [
    "albumentations",
    "av",
    "catboost",
    "cv2",
    "dask.dataframe",
    "easyocr",
    "faiss",
    "fastai",
    "h5py",
    "imagecodecs",
    "imageio",
    "kornia",
    "lightgbm",
    "lmdb",
    "matplotlib",
    "monai",
    "nibabel",
    "numpy",
    "onnx",
    "onnxruntime",
    "openslide",
    "optuna",
    "pandas",
    "PIL",
    "plotly",
    "polars",
    "pyarrow",
    "pycocotools",
    "pydicom",
    "pytesseract",
    "pyvips",
    "rasterio",
    "rioxarray",
    "scipy",
    "segmentation_models_pytorch",
    "shap",
    "SimpleITK",
    "skimage",
    "sklearn",
    "statsmodels",
    "tables",
    "tifffile",
    "timm",
    "torch",
    "torchmetrics",
    "torchvision",
    "transformers",
    "webdataset",
    "xarray",
    "xgboost",
    "zarr",
]


def import_smoke() -> None:
    failures: list[str] = []
    for module in IMPORTS:
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
                capture_output=True,
                check=False,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{module}: import exceeded 90 seconds")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            failures.append(f"{module}: exit {result.returncode}: {detail}")
    if failures:
        raise RuntimeError("Direct dependency import failures:\n" + "\n".join(failures))


def fixture_smoke() -> None:
    import numpy as np
    import onnx
    import pandas as pd
    import pyarrow as pa
    import rasterio
    import torch
    from PIL import Image
    from rasterio.io import MemoryFile

    frame = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    assert pa.Table.from_pandas(frame).num_rows == 2
    print("fixture: table", flush=True)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, "PNG")
    assert Image.open(io.BytesIO(buffer.getvalue())).size == (8, 8)
    print("fixture: image", flush=True)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=2, height=2, count=1, dtype="uint8"
        ) as dataset:
            dataset.write(np.ones((1, 2, 2), dtype=np.uint8))
        with memory.open() as dataset:
            assert dataset.read(1).sum() == 4
    print("fixture: raster", flush=True)
    model = torch.nn.Linear(2, 1).eval()
    onnx_path = "/tmp/imperium-smoke.onnx"
    torch.onnx.export(model, torch.ones(1, 2), onnx_path)
    print("fixture: onnx export", flush=True)
    onnx.checker.check_model(onnx.load(onnx_path))
    assert rasterio.__version__
    manifest = json.loads(Path("/opt/imperium/weights-manifest.json").read_text())
    assert manifest == {"schema_version": 1, "weights": []}


def gpu_smoke() -> None:
    import torch
    from torchvision.ops import nms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    product = left @ right
    assert torch.isfinite(product).all().item()
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    assert nms(boxes, scores, 0.5).numel() == 1
    torch.cuda.synchronize()
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}))


def dali_smoke() -> None:
    from nvidia.dali import fn, pipeline_def, types
    from PIL import Image

    image_root = Path("/tmp/dali-smoke")
    image_root.mkdir(parents=True, exist_ok=True)
    files = []
    for index, color in enumerate(((20, 40, 60), (80, 100, 120))):
        path = image_root / f"{index}.jpg"
        Image.new("RGB", (24, 18), color).save(path, "JPEG")
        files.append(str(path))

    @pipeline_def(batch_size=2, num_threads=2, device_id=0)
    def pipeline():
        encoded, labels = fn.readers.file(files=files, random_shuffle=False)
        decoded = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
        resized = fn.resize(decoded, device="gpu", resize_x=16, resize_y=16)
        normalized = fn.crop_mirror_normalize(
            resized,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return normalized, labels

    pipe = pipeline()
    pipe.build()
    outputs = pipe.run()
    assert outputs[0].as_cpu().as_array().shape == (2, 3, 16, 16)
    assert outputs[1].as_array().shape == (2, 1)
    print("dali: jpeg decode, resize, normalize ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--dali", action="store_true")
    args = parser.parse_args()
    if not args.gpu:
        import_smoke()
        print("imports: ok", flush=True)
        fixture_smoke()
        print("fixtures: ok", flush=True)
    if args.gpu:
        gpu_smoke()
    if args.dali:
        dali_smoke()


if __name__ == "__main__":
    main()
