from __future__ import annotations

import math
from dataclasses import dataclass


STIMULUS_BLOCKS = (
    ("grating", "stimulus"),
    ("stop_after_grating", "no_stimulus"),
    ("dot", "stimulus"),
    ("static", "no_stimulus"),
)


@dataclass(frozen=True)
class FrameTimingLabel:
    index: int
    cycle: int
    center_s: float
    center_in_cycle_s: float
    block_start_s: float
    block_name: str
    binary_label: str


def infer_visual_stimulus_label(
    index: int,
    group_seconds: float = 4.0,
    cycle_seconds: float = 120.0,
) -> FrameTimingLabel:
    """Infer visual-stimulus labels from file index and the PPT timing notes."""
    zero_based = index - 1
    start_s = zero_based * group_seconds
    center_s = start_s + group_seconds / 2.0
    cycle = math.floor(start_s / cycle_seconds)
    center_in_cycle_s = center_s % cycle_seconds
    block_i = min(int(center_in_cycle_s // 30.0), len(STIMULUS_BLOCKS) - 1)
    block_start_s = block_i * 30.0
    block_name, binary_label = STIMULUS_BLOCKS[block_i]
    return FrameTimingLabel(
        index=index,
        cycle=cycle,
        center_s=center_s,
        center_in_cycle_s=center_in_cycle_s,
        block_start_s=block_start_s,
        block_name=block_name,
        binary_label=binary_label,
    )


def is_clean_block_middle(label: FrameTimingLabel, margin_s: float = 8.0) -> bool:
    """Keep the central part of each 30 s block, away from transition edges."""
    position = label.center_in_cycle_s - label.block_start_s
    return margin_s <= position <= (30.0 - margin_s)
