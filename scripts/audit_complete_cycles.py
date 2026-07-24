#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.datasets import load_monkey_session


def main() -> None:
    rows = []
    data_root = PROJECT_DIR / "data"
    for session_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        session = session_dir.name
        _, _, _, meta = load_monkey_session(
            PROJECT_DIR,
            session=session,
            task="binary",
            clean_middle=False,
            analysis_limit=None,
            trim_incomplete_cycles=True,
        )
        info = meta.attrs["selection_info"]
        incomplete = info["incomplete_cycles_after_analysis_limit"]
        rows.append(
            {
                "session": session,
                "raw_frame_count": info["raw_frame_count"],
                "frames_per_cycle_expected": info["frames_per_cycle_expected"],
                "frames_after_analysis_limit": info["frames_after_analysis_limit"],
                "complete_cycles_after_analysis_limit": info["n_complete_cycles_after_analysis_limit"],
                "frames_after_complete_cycle_trim": info["frames_after_complete_cycle_trim"],
                "frames_dropped_incomplete_cycles": info["frames_dropped_incomplete_cycles"],
                "indices_dropped_incomplete_cycles": ",".join(
                    str(index) for index in info["indices_dropped_incomplete_cycles"]
                ),
                "incomplete_cycles": json.dumps(incomplete, ensure_ascii=False),
            }
        )

    out_dir = PROJECT_DIR / "reports" / "decoding" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "complete_cycle_audit.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved complete-cycle audit to {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
