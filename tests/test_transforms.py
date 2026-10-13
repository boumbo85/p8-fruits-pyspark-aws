"""
Unit tests for the pure transforms (no Spark, no TensorFlow, no cluster).

These cover the two bits of logic that are easy to get wrong and don't need a
GPU to verify: deriving the label from the image path, and decoding/scaling an
image to the model's input format.
"""

import io

import numpy as np
import pytest
from PIL import Image

from transforms import TARGET_SIZE, label_from_path, preprocess_image

# ── label_from_path ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "path, expected",
    [
        ("s3://bucket/data/Test/Apple/img_01.jpg", "Apple"),
        ("s3://bucket/data/Test/Banana_1/x.jpg", "Banana_1"),
        ("/local/data/Pear/0.jpg", "Pear"),
    ],
)
def test_label_from_path(path, expected):
    assert label_from_path(path) == expected


def test_label_from_path_rejects_too_shallow():
    with pytest.raises(ValueError):
        label_from_path("img.jpg")


# ── preprocess_image ─────────────────────────────────────────────

def _png_bytes(width, height, color=(120, 200, 50)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_image_shape_is_model_input():
    arr = preprocess_image(_png_bytes(100, 100))
    assert arr.shape == (TARGET_SIZE[0], TARGET_SIZE[1], 3)


def test_preprocess_image_resizes_non_square_input():
    arr = preprocess_image(_png_bytes(640, 480))
    assert arr.shape == (224, 224, 3)


def test_preprocess_image_is_scaled_to_minus_one_one():
    arr = preprocess_image(_png_bytes(10, 10, color=(255, 255, 255)))
    assert arr.dtype == np.float32
    assert arr.max() <= 1.0 and arr.min() >= -1.0
    # pure white (255) must map to +1.0
    assert np.allclose(arr, 1.0)


def test_preprocess_image_black_maps_to_minus_one():
    arr = preprocess_image(_png_bytes(10, 10, color=(0, 0, 0)))
    assert np.allclose(arr, -1.0)
