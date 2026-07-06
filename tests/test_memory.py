"""Memory utility tests."""

from lfm25_ja.utils.memory import format_bytes, get_vram_usage


def test_format_bytes() -> None:
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(0) == "0 B"


def test_get_vram_usage_returns_dict() -> None:
    usage = get_vram_usage()
    assert "allocated" in usage
    assert "reserved" in usage
    assert usage["allocated"] >= 0
