"""
test_dataset.py - Unit test that runs data/verify_dataset.py checks.
"""
from data.verify_dataset import verify


def test_canonical_dataset_verification():
    assert verify() is True
