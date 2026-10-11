"""Sample 1/N of a JSONL file without replacement, then repeat it N times.

Examples:
    python repeat_dolma.py --repeat 2
    python repeat_dolma.py --repeat 4
    python repeat_dolma.py --sample-fraction 0.4 --output dolma3_100gb.jsonl

Selects exactly floor(record_count / N) records with seed 42 by default.
Selected records retain source order; the output contains N identical passes.
Uses bounded memory and only the output's .part file for temporary storage.
--sample-fraction selects that fraction of records once, without repetition.
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


CHUNK_SIZE = 8 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dolma3_250gb.jsonl"))
    parser.add_argument("--output", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--repeat", type=int)
    selection.add_argument("--sample-fraction", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.repeat = args.repeat if args.repeat is not None else (1 if args.sample_fraction is not None else 2)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.sample_fraction is not None and not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be in (0, 1]")

    source = args.input
    suffix = f"sample_{args.sample_fraction:g}" if args.sample_fraction is not None else f"repeat_{args.repeat}"
    output = args.output or source.with_name(f"{source.stem}_{suffix}.jsonl")
    partial = output.with_name(output.name + ".part")
    for path in (output, partial):
        if path.exists():
            parser.error(f"{path} already exists; refusing to overwrite")

    initial_stat = source.stat()

    def check_source():
        current = source.stat()
        if (current.st_size, current.st_mtime_ns) != (
            initial_stat.st_size, initial_stat.st_mtime_ns
        ):
            raise RuntimeError("Source changed while processing; output was not published")

    total = 0
    last_byte = b""
    with source.open("rb") as src, tqdm(
        total=initial_stat.st_size, desc="Counting records", unit="B", unit_scale=True
    ) as progress:
        while chunk := src.read(CHUNK_SIZE):
            total += chunk.count(b"\n")
            last_byte = chunk[-1:]
            progress.update(len(chunk))
    if last_byte and last_byte != b"\n":
        total += 1
    check_source()
    sample_count = int(total * args.sample_fraction) if args.sample_fraction is not None else total // args.repeat
    if not sample_count:
        parser.error("Requested sample contains no records")
    print(
        f"Sampling {sample_count:,} of {total:,} records (seed {args.seed}); "
        f"output will contain {sample_count * args.repeat:,} records.",
        flush=True,
    )

    rng = random.Random(args.seed)
    needed = sample_count
    text_bytes = 0
    with source.open("rb") as src, partial.open("xb", buffering=CHUNK_SIZE) as dst, tqdm(
        total=sample_count, desc="Sampling records", unit=" records", unit_scale=True
    ) as progress:
        for index, line in enumerate(src):
            # Conditional selection gives a uniform subset of exactly sample_count.
            if rng.randrange(total - index) < needed:
                if args.sample_fraction is not None:
                    text_bytes += len(json.loads(line)["text"].encode("utf-8"))
                dst.write(line)
                if not line.endswith(b"\n"):
                    dst.write(b"\n")
                needed -= 1
                progress.update(1)
            if needed == 0:
                break
    if needed:
        raise RuntimeError("Source ended before the sample was complete")
    check_source()

    sample_bytes = partial.stat().st_size
    with partial.open("ab", buffering=CHUNK_SIZE) as dst:
        for repetition in range(2, args.repeat + 1):
            with partial.open("rb") as src, tqdm(
                total=sample_bytes,
                desc=f"Writing repetition {repetition}/{args.repeat}",
                unit="B",
                unit_scale=True,
            ) as progress:
                remaining = sample_bytes
                while remaining:
                    chunk = src.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        raise RuntimeError("Unexpected end of sampled data")
                    dst.write(chunk)
                    remaining -= len(chunk)
                    progress.update(len(chunk))
                dst.flush()

    if partial.stat().st_size != sample_bytes * args.repeat:
        raise RuntimeError("Output size does not match the requested repetitions")
    partial.replace(output)
    print(f"Saved {output} ({sample_bytes * args.repeat:,} bytes)", flush=True)
    if args.sample_fraction is not None:
        print(f"Sample contains {sample_count:,} records, {text_bytes:,} UTF-8 text bytes ({text_bytes / 1e9:.6f} GB).", flush=True)


if __name__ == "__main__":
    main()
