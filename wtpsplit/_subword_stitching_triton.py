"""Round ordered subword overlap-add at the same points as the NumPy path."""
from importlib import import_module
from typing import Any

triton: Any = import_module("triton")
tl: Any = import_module("triton.language")


@triton.jit  # type: ignore[untyped-decorator]
def overlap_add(
    RAW: tl.tensor, WINDOWS: tl.tensor, TILES: tl.tensor, WEIGHTS: tl.tensor, OUT: tl.tensor,
    WIDTH: tl.constexpr, LABELS: tl.constexpr, CLS: tl.constexpr, HAT: tl.constexpr, TILE: tl.constexpr,
) -> None:
    tile, label = tl.program_id(0), tl.program_id(1)
    start = tl.load(TILES + tile * 4)
    stop = tl.load(TILES + tile * 4 + 1)
    first = tl.load(TILES + tile * 4 + 2).to(tl.int32)
    last = tl.load(TILES + tile * 4 + 3).to(tl.int32)
    pos = start + tl.arange(0, TILE)
    valid = pos < stop

    if not HAT and tl.load(WINDOWS + first * 4 + 2) == 0:
        offset = pos - tl.load(WINDOWS + first * 4)
        source = tl.load(WINDOWS + first * 4 + 1)
        value = tl.load(RAW + ((source * WIDTH + offset + CLS) * LABELS + label), valid, 0)
        tl.store(OUT + pos * LABELS + label, value, valid)
    else:
        acc = tl.full((TILE,), 0, tl.float16)
        count = tl.full((TILE,), 0, tl.float16)
        for row in range(first, last):
            offset = pos - tl.load(WINDOWS + row * 4)
            source = tl.load(WINDOWS + row * 4 + 1)
            value = tl.load(RAW + ((source * WIDTH + offset + CLS) * LABELS + label), valid, 0)
            if HAT:
                weight = tl.load(WEIGHTS + offset, valid, 0)
                value = (value.to(tl.float32) * weight.to(tl.float32)).to(value.dtype)
                count = (count.to(tl.float32) + weight.to(tl.float32)).to(tl.float16)
            next_acc = (acc.to(tl.float32) + value.to(tl.float32)).to(tl.float16)
            if not HAT and RAW.dtype.element_ty == tl.float16:
                # The host copies the initial FP16 window if it starts at zero.
                # Preserve its signed zero when this is the only contribution.
                acc = tl.where(tl.load(WINDOWS + row * 4 + 3) != 0, value, next_acc)
            else:
                acc = next_acc
        if not HAT:
            count = (last - first).to(tl.float16)
        value = tl.div_rn(acc.to(tl.float32), tl.where(valid, count, 1.0).to(tl.float32))
        tl.store(OUT + pos * LABELS + label, value, valid)
