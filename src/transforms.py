"""
Pure, dependency-light transforms used by the Spark pipeline.

Kept free of TensorFlow and PySpark on purpose so they can be unit-tested in CI
without a cluster or a GPU. ``featurize_pca.py`` imports from here; the heavy
Spark/TensorFlow wiring lives there.
"""

import io

import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)


def label_from_path(path: str) -> str:
    """
    Derive the class label from an image path — the parent folder name.

    Mirrors the Spark expression ``element_at(split(path, '/'), -2)`` used in the
    pipeline, so the label logic is testable without Spark.

    >>> label_from_path("s3://bucket/data/Test/Apple/img_01.jpg")
    'Apple'
    """
    parts = [p for p in path.split("/") if p != ""]
    if len(parts) < 2:
        raise ValueError(f"path has no parent folder to use as label: {path!r}")
    return parts[-2]


def preprocess_image(content: bytes) -> np.ndarray:
    """
    Decode raw image bytes and resize to the model input size.

    Returns a float32 array of shape (224, 224, 3) scaled to [-1, 1] — the same
    scaling MobileNetV2's ``preprocess_input`` applies — reimplemented in NumPy
    so the step needs neither TensorFlow nor a GPU to run or test.
    """
    img = Image.open(io.BytesIO(content)).convert("RGB").resize(TARGET_SIZE)
    arr = np.asarray(img, dtype="float32")
    # MobileNetV2 'tf' mode: map [0, 255] -> [-1, 1]
    return arr / 127.5 - 1.0
