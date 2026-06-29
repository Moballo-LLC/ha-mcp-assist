"""Tests for release validation helper scripts."""

from __future__ import annotations

from pathlib import Path


def test_release_candidate_ancestry_fetch_preserves_main_history() -> None:
    script = Path("scripts/verify_release_candidate.sh").read_text(encoding="utf-8")
    function_start = script.index("fetch_main_for_ancestry_check()")
    function_end = script.index('if [[ "${REQUIRE_MAIN_ANCESTOR:-0}" == "1" ]]')
    fetch_function = script[function_start:function_end]
    require_main_block = script[function_end : script.index('if [[ "${SKIP_LOCAL_VERIFY:-0}"')]

    assert "--depth" not in fetch_function
    assert "--unshallow" in fetch_function
    assert "+refs/heads/main:refs/remotes/origin/main" in fetch_function
    assert "fetch_main_for_ancestry_check" in require_main_block
    assert "git merge-base --is-ancestor HEAD origin/main" in require_main_block
