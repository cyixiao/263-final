#!/usr/bin/env python3
"""Build the analytic MIMIC-IV demo dataset used by all model scripts."""

from __future__ import annotations

import json
from pathlib import Path

import importlib.util


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "archive" / "raw_mimic_demo"


def load_legacy_builder():
    """Reuse the validated cohort/feature code while model scripts are split out."""
    path = ROOT / "archive" / "script_old" / "analyze_mimic_demo.py"
    spec = importlib.util.spec_from_file_location("mimic_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    DATA.mkdir(exist_ok=True, parents=True)
    builder = load_legacy_builder()
    builder.DATA = RAW
    analytic, meta = builder.build_dataset()
    table1 = builder.table_one(analytic)

    analytic.to_csv(DATA / "analytic.csv", index=False)
    table1.to_csv(DATA / "table1.csv", index=False)
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps({**meta, "analytic": str(DATA / "analytic.csv")}, indent=2))


if __name__ == "__main__":
    main()
