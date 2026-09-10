"""Optional, file-backed cache of exactly the original normalized frames."""

from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_identity(dataset_root: Path) -> dict:
    from .dataset import IMAGENET_MEAN, IMAGENET_STD, MiniWAMDataset

    digest = hashlib.sha256()
    paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not paths:
        raise ValueError("No dataset parquet files")
    for path in paths:
        digest.update(str(path.relative_to(dataset_root)).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "version": 1,
        "dataset_sha256": digest.hexdigest(),
        "normalizer_sha256": hashlib.sha256(
            inspect.getsource(MiniWAMDataset._normalize_image).encode()
        ).hexdigest(),
        "mean": IMAGENET_MEAN.flatten().tolist(),
        "std": IMAGENET_STD.flatten().tolist(),
        "versions": {name: version(name) for name in
                     ("torch", "torchvision", "lerobot", "datasets", "Pillow")},
    }


class NormalizedImageCache:
    """Memory-map frames; reopen after process spawning instead of pickling data."""

    def __init__(self, path: str | Path, dataset_root: Path, frame_count: int):
        self.path = Path(path).resolve()
        manifest = json.loads((self.path / "manifest.json").read_text())
        if manifest["identity"] != cache_identity(dataset_root):
            raise ValueError("Image cache does not match dataset/normalization")
        self.shape = (frame_count, 3, 96, 96)
        if tuple(manifest["shape"]) != self.shape:
            raise ValueError("Image cache frame count/shape mismatch")
        if (self.path / "frames.npy").stat().st_size != manifest["file_bytes"]:
            raise ValueError("Image cache file is incomplete")
        if file_digest(self.path / "frames.npy") != manifest["frames_sha256"]:
            raise ValueError("Image cache checksum mismatch")
        self._array = None
        self._open()

    def _open(self):
        if self._array is None:
            # Copy-on-write mappings share file-backed pages. Returned samples
            # are stacked into new tensors, never mutated in this mapping.
            self._array = np.load(self.path / "frames.npy", mmap_mode="c")
            if self._array.shape != self.shape or self._array.dtype != np.float32:
                raise ValueError("Image cache array shape/dtype mismatch")

    def __getstate__(self):
        return {**self.__dict__, "_array": None}

    def __getitem__(self, index: int) -> torch.Tensor:
        self._open()
        return torch.from_numpy(self._array[index])


def build_image_cache(dataset, destination: str | Path) -> Path:
    """Build once using the existing reader and normalization, then publish."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Cache already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        count = len(dataset._states)
        shape = (count, 3, 96, 96)
        identity = cache_identity(dataset.dataset_root)
        array = np.lib.format.open_memmap(
            temporary / "frames.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for index in range(count):
            row = dataset.source[index]
            if int(row["index"]) != index:
                raise ValueError("Source frame index is not contiguous")
            image = dataset._normalize_image(row["observation.image"])
            if image.shape != shape[1:] or image.dtype != torch.float32:
                raise ValueError("Unexpected normalized image contract")
            array[index] = image.numpy()
            if not np.array_equal(array[index], image.numpy()):
                raise ValueError("Cache write changed a normalized frame")
            if (index + 1) % 2000 == 0:
                print(f"image_cache frames={index + 1}/{count}", flush=True)
        array.flush()
        del array
        if cache_identity(dataset.dataset_root) != identity:
            raise ValueError("Source dataset changed during cache construction")
        manifest = {"identity": identity, "shape": shape,
                    "file_bytes": (temporary / "frames.npy").stat().st_size,
                    "frames_sha256": file_digest(temporary / "frames.npy")}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2))
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination
