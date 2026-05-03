"""
tests/conftest.py
=================
Pytest fixtures shared across the test suite.

The `model` fixture loads YOLOv8m once per test module and yields it
to any test that takes `model` as a parameter. Module-scoped because
loading the checkpoint takes ~1s and the tests don't mutate the model.

Used by: tests/test_canonical_infer.py
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def model():
    """YOLOv8m model, loaded once per test module."""
    from ultralytics import YOLO
    return YOLO("yolov8m.pt")