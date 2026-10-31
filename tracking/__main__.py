"""CLI entry point: python -m tracking"""

import argparse
import sys


def cli():
    p = argparse.ArgumentParser(prog="openmotion", description="OpenMotion SDK")
    p.add_argument("--detect", action="store_true", help="List detected sensors")
    p.add_argument("--viewer", action="store_true", help="Launch the viewer UI")
    p.add_argument("--version", action="store_true", help="Print version")
    args = p.parse_args()

    if args.version:
        from . import __version__
        print(f"openmotion {__version__}")
        return

    if args.detect:
        from .detect import main as detect_main
        detect_main()
        return

    # Default: launch viewer
    from .live_viewer import main as viewer_main
    viewer_main()


if __name__ == "__main__":
    cli()
