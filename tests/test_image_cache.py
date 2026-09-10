from pathlib import Path
import pickle

import pytest
import torch

from mini_wam.data.dataset import MiniWAMDataset, NormalizationStats
from mini_wam.data.image_cache import build_image_cache, NormalizedImageCache


@pytest.fixture
def original(tmp_path):
    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    (root / "data" / "test.parquet").write_bytes(b"synthetic dataset identity")
    data = object.__new__(MiniWAMDataset)
    data.dataset_root = root
    data.normalization = NormalizationStats(torch.zeros(2), torch.ones(2),
                                           torch.ones(2), torch.full((2,), 2.0))
    generator = torch.Generator().manual_seed(42)
    data.source = [{"index": torch.tensor(i), "observation.image":
                    torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8,
                                  generator=generator)} for i in range(6)]
    data._states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    data._actions = data._states + 10
    data._windows = [(0, i, i, 6) for i in range(1, 5)]
    data._image_cache = None
    data.action_horizon = 16
    data.future_horizon = 4
    data.include_future_observations = True
    return data


def test_cached_windows_exact_and_samples_cannot_mutate_cache(original, tmp_path):
    expected = [original[i] for i in range(len(original))]
    before_rng = torch.get_rng_state().clone()
    path = build_image_cache(original, tmp_path / "cache")
    original._image_cache = NormalizedImageCache(path, original.dataset_root, 6)
    assert torch.equal(before_rng, torch.get_rng_state())
    for i, sample in enumerate(expected):
        actual = original[i]
        assert actual.keys() == sample.keys()
        for key in sample:
            assert torch.equal(actual[key], sample[key]), key
        actual["future_observations"].zero_()
        assert torch.equal(original[i]["future_observations"], sample["future_observations"])
    original.include_future_observations = False
    assert "future_observations" not in original[0]
    assert torch.equal(original[0]["observation_history"], expected[0]["observation_history"])
    restored = pickle.loads(pickle.dumps(original._image_cache))
    assert restored._array is None
    assert torch.equal(restored[2], original._image_cache[2])
    assert len(pickle.dumps(restored)) < 2000


def test_cache_rejects_changed_source_and_corruption(original, tmp_path):
    path = build_image_cache(original, tmp_path / "cache")
    with pytest.raises(FileExistsError):
        build_image_cache(original, path)
    with pytest.raises(ValueError, match="frame count"):
        NormalizedImageCache(path, original.dataset_root, 5)
    source = original.dataset_root / "data" / "test.parquet"
    before = source.read_bytes()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        NormalizedImageCache(path, original.dataset_root, 6)
    source.write_bytes(before)
    with (path / "frames.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="checksum"):
        NormalizedImageCache(path, original.dataset_root, 6)


def test_cache_rejects_frame_reordering_without_publishing(original, tmp_path):
    original.source[2]["index"] = torch.tensor(1)
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="not contiguous"):
        build_image_cache(original, target)
    assert not target.exists()
    assert not list(tmp_path.glob("cache.tmp-*"))
