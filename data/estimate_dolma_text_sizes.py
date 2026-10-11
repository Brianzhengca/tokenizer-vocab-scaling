"""Estimate text bytes by sampling three complete shards per source/topic.

Run with the pinned Hugging Face tree inventory as the only argument.
Completed shard measurements are checkpointed for restart.
"""

import concurrent.futures
import csv
import io
import json
import random
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import zstandard


def measure(entry, revision):
    url = (
        "https://huggingface.co/datasets/allenai/dolma3_mix-6T/resolve/"
        + revision + "/" + urllib.parse.quote(entry["path"])
    )
    for attempt in range(3):
        try:
            with tempfile.TemporaryFile() as compressed:
                with urllib.request.urlopen(url, timeout=120) as response:
                    while block := response.read(4 * 1024 * 1024):
                        compressed.write(block)
                if compressed.tell() != entry["size"]:
                    raise ValueError("Downloaded size differs from inventory")
                compressed.seek(0)
                text_bytes = records = 0
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    with io.TextIOWrapper(reader, encoding="utf-8") as lines:
                        for line in lines:
                            record = json.loads(line)
                            text_bytes += len(record["text"].encode("utf-8"))
                            records += 1
            return dict(entry, text_bytes=text_bytes, records=records, revision=revision)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    inventory = json.loads(Path(sys.argv[1]).read_text())
    revision = inventory["revision"]
    groups = defaultdict(list)
    for entry in inventory["entries"]:
        if entry["type"] == "file" and entry["path"].endswith(".zst"):
            topic = re.sub(r"-\d+$", "", entry["path"].split("/")[1])
            topic = re.sub(r"-part\d+$", "", topic)
            groups[topic].append(entry)
    rng = random.Random(42)
    selected = {
        topic: rng.sample(sorted(entries, key=lambda e: e["path"]), min(3, len(entries)))
        for topic, entries in sorted(groups.items())
    }
    checkpoint = Path("dolma_sampled_shards.jsonl")
    completed = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            result = json.loads(line)
            if result["revision"] == revision:
                completed[result["path"]] = result
    pending = [e for entries in selected.values() for e in entries if e["path"] not in completed]
    print(f"Measuring {len(pending)} shards, {sum(e['size'] for e in pending)/1e9:.2f} compressed GB", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool, checkpoint.open("a") as log:
        futures = {pool.submit(measure, entry, revision): entry for entry in pending}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed[result["path"]] = result
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(f"Measured {len(completed)}: {result['path']}", flush=True)
    rows = []
    for topic, entries in sorted(groups.items()):
        samples = [completed[e["path"]] for e in selected[topic]]
        compressed_total = sum(e["size"] for e in entries)
        sampled_compressed = sum(e["size"] for e in samples)
        sampled_text = sum(e["text_bytes"] for e in samples)
        ratio = sampled_text / sampled_compressed
        ratios = [e["text_bytes"] / e["size"] for e in samples]
        rows.append(dict(source_topic=topic, shards=len(entries), sampled_shards=len(samples),
                         compressed_bytes=compressed_total, sampled_compressed_bytes=sampled_compressed,
                         sampled_text_bytes=sampled_text, sampled_documents=sum(e["records"] for e in samples),
                         text_to_compressed_ratio=ratio, min_sample_ratio=min(ratios), max_sample_ratio=max(ratios),
                         estimated_text_bytes=compressed_total * ratio))
    total = sum(r["estimated_text_bytes"] for r in rows)
    for row in rows:
        row["estimated_text_GB"] = row["estimated_text_bytes"] / 1e9
        row["estimated_text_share_percent"] = 100 * row["estimated_text_bytes"] / total
        row["target_text_GB_for_250GB"] = 250 * row["estimated_text_bytes"] / total
    with open("dolma_topic_text_estimates.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = defaultdict(float)
    for row in rows:
        sources[row["source_topic"].split("-", 1)[0]] += row["estimated_text_bytes"]
    with open("dolma_source_text_estimates.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "estimated_text_GB", "estimated_text_share_percent", "target_text_GB_for_250GB"])
        for source, size in sorted(sources.items(), key=lambda p: -p[1]):
            writer.writerow([source, size/1e9, 100*size/total, 250*size/total])
    metadata = dict(dataset="allenai/dolma3_mix-6T", revision=revision, seed=42,
                    method="Simple random sample without replacement of up to 3 complete shards per source/topic. Estimated topic text bytes = total compressed bytes * sampled text bytes / sampled compressed bytes (ratio estimator).",
                    text_definition="UTF-8 bytes of the text field only; excludes metadata and JSON serialization overhead.",
                    limitations="Small pilot sample; ratio estimates may be biased and variable. Sample min/max ratios are diagnostics, not confidence intervals. No token counts inferred.",
                    sampled_shards=sum(len(v) for v in selected.values()),
                    sampled_compressed_bytes=sum(r["sampled_compressed_bytes"] for r in rows),
                    estimated_total_text_bytes=total)
    Path("dolma_text_estimate_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
