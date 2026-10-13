#!/usr/bin/env python3
"""Keep the Embedded Python Blocks inside grc/respiratory_radar_V4.grc equal to
respradar/blocks/*.py.

GRC stores a copy of each block's source in the .grc file. This tool loads the
flowgraph through GRC's own API, matches each embedded block to a module in
respradar/blocks by the block name declared in its gr.sync_block.__init__,
replaces the embedded source, and saves the file back in GRC's format.

    python tools/sync_grc_blocks.py            # update the .grc
    python tools/sync_grc_blocks.py --check    # exit 1 if anything differs
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRC_FILE = os.path.join(REPO, "grc", "respiratory_radar_V4.grc")
BLOCK_DIR = os.path.join(REPO, "respradar", "blocks")


def load_platform():
    from gnuradio import gr
    from gnuradio.grc.core.platform import Platform
    platform = Platform(name="sync_grc_blocks", prefs=gr.prefs(), version=gr.version(),
                        version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()))
    platform.build_library()
    return platform


def package_blocks():
    """{block name declared in the source: source text} for respradar/blocks/*.py."""
    from gnuradio.grc.core.utils import epy_block_io
    found = {}
    for fn in sorted(os.listdir(BLOCK_DIR)):
        if fn.startswith("_") or not fn.endswith(".py"):
            continue
        with open(os.path.join(BLOCK_DIR, fn), encoding="utf-8") as f:
            src = f.read()
        found[epy_block_io.extract(src).name] = src
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report differences, change nothing")
    ap.add_argument("--grc", default=GRC_FILE)
    args = ap.parse_args(argv)

    platform = load_platform()
    fg = platform.make_flow_graph(args.grc)
    sources = package_blocks()

    changed, unmatched = [], []
    for blk in fg.blocks:
        if blk.key != "epy_block":
            continue
        from gnuradio.grc.core.utils import epy_block_io
        embedded = blk.params["_source_code"].get_value()
        name = epy_block_io.extract(embedded).name
        if name not in sources:
            unmatched.append((blk.name, name))
            continue
        if embedded != sources[name]:
            changed.append((blk.name, name))
            blk.params["_source_code"].set_value(sources[name])

    for blk_id, name in unmatched:
        print(f"no module in respradar/blocks declares a block named {name!r} (used by {blk_id})")
    for blk_id, name in changed:
        print(f"{'differs' if args.check else 'updated'}: {blk_id} ({name})")
    if unmatched:
        return 2
    if args.check:
        print("in sync" if not changed else f"{len(changed)} block(s) out of sync")
        return 1 if changed else 0
    if changed:
        fg.rewrite()
        fg.validate()
        platform.save_flow_graph(args.grc, fg)
        print(f"saved {os.path.relpath(args.grc, REPO)}")
    else:
        print("in sync, nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
