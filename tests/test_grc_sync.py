"""The flowgraph's embedded block sources must match respradar/blocks/*.py."""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_grc_embedded_blocks_match_package():
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "sync_grc_blocks.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_grc_compiles_with_grcc(tmp_path):
    """grcc evaluates every parameter expression, so this catches namespace errors
    like the missing `window` import the delivered file had."""
    r = subprocess.run(["grcc", "-o", str(tmp_path), os.path.join(REPO, "grc", "respiratory_radar_V4.grc")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "respiratory_radar.py").exists()
