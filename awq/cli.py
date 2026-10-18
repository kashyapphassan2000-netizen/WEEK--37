"""cli.py — the shared command-line flags every script accepts: --config and --set key=value."""
from __future__ import annotations

import argparse

from .config import load_config


def parser(description: str, need_config: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=need_config, help="path to configs/run.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set quant.group_size=64 (repeatable)")
    return p


def run_config(args):
    return load_config(args.config, overrides=args.set)
