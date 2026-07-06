"""Package scaffold tests."""

import importlib

import lfm25_ja


def test_import_package() -> None:
    assert importlib.import_module("lfm25_ja") is lfm25_ja


def test_version_defined() -> None:
    assert isinstance(lfm25_ja.__version__, str)
    assert lfm25_ja.__version__
