#!/usr/bin/env python3
"""Prepare the 12 external evaluation datasets used by the FediAID paper.

This script converts public source files into the five seeded JSONL layouts used
by the reproduction runner. It does not redistribute any source dataset.
Use --encode to call src.encode_posts.py and create the precomputed-embedding
variants expected by scripts/run_opensrc_reproduction.py.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import subprocess
import sys
from pathlib import Path

SEEDS = (0, 1, 2, 3, 4)
TEST_FRACTION = 0.2
PLATFORMS = ("medium", "quora", "reddit")
MULTISOCIAL_PLATFORMS = ("discord", "gab", "telegram", "twitter", "whatsapp")


def clean_text(raw: object) -> str:
    text = "" if raw is None else str(raw)
    text = re.sub(r"^(['\"])(.*)\1$", r"\2", text)
    text = text.replace("<br>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def seeded_test_records(records: list[dict], seed: int, inplace: bool = False) -> list[dict]:
    shuffled = records if inplace else records.copy()
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_FRACTION))
    return shuffled[:n_test]


def prepare_aigtbench(out_root: Path, seeds: list[int]) -> None:
    from datasets import load_dataset

    dataset = load_dataset("tarryzhang/AIGTBench")
    records_by_platform = {platform: [] for platform in PLATFORMS}
    for split in ("train", "test"):
        for item in dataset[split]:
            platform = str(item.get("social_media_platform", "")).lower()
            if platform not in records_by_platform:
                continue
            text = clean_text(item.get("text"))
            if text:
                records_by_platform[platform].append({"text": text, "label": int(item["label"])})
    for seed in seeds:
        for platform, records in records_by_platform.items():
            path = out_root / f"opensrc_platforms_seed{seed}" / f"AIGTB_{platform}.jsonl"
            write_jsonl(path, seeded_test_records(records, seed))


def prepare_multisocial(csv_path: Path, out_root: Path, seeds: list[int]) -> None:
    import pandas as pd

    frame = pd.read_csv(csv_path)
    frame = frame.loc[frame["potential_noise"] == 0].copy()
    frame["platform"] = frame["source"].map(lambda value: str(value).replace("multisocial_", ""))
    for seed in seeds:
        for platform in MULTISOCIAL_PLATFORMS:
            subset = frame.loc[frame["platform"] == platform].sample(frac=1, random_state=seed)
            n_test = max(1, int(len(subset) * TEST_FRACTION))
            records = []
            for row in subset.iloc[:n_test].itertuples(index=False):
                text = clean_text(getattr(row, "text"))
                if text:
                    records.append({"text": text, "label": int(getattr(row, "label"))})
            write_jsonl(out_root / f"opensrc_platforms_seed{seed}" / f"multisocial_{platform}.jsonl", records)


def prepare_fox8(path: Path, out_root: Path, seeds: list[int]) -> None:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            label = 0 if item["label"] == "human" else 1
            for tweet in item["user_tweets"]:
                text = clean_text(tweet.get("text"))
                if text:
                    records.append({"text": text, "label": label})
    for seed in seeds:
        test = seeded_test_records(records, seed, inplace=True)
        write_jsonl(out_root / f"opensrc_platforms_seed{seed}" / "fox8.jsonl", test)


def prepare_tweepfake(splits_dir: Path, out_root: Path, seeds: list[int]) -> None:
    import pandas as pd

    paths = [splits_dir / name for name in ("train.csv", "validation.csv", "test.csv")]
    frame = pd.concat([pd.read_csv(path, delimiter=";") for path in paths], ignore_index=True)
    records = []
    for _, row in frame.iterrows():
        text = clean_text(row.get("text"))
        if text:
            records.append({"text": text, "label": int(row.get("account.type") != "human")})
    for seed in seeds:
        test = seeded_test_records(records, seed, inplace=True)
        write_jsonl(out_root / f"opensrc_platforms_seed{seed}" / "tweepfake.jsonl", test)


def prepare_m4(source_dir: Path, out_root: Path, seeds: list[int]) -> None:
    records = []
    for path in sorted(source_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for item in (json.loads(line) for line in stream if line.strip()):
                if "machine_answer" in item and "title" in item:
                    human_text, generated_text = item.get("text"), item.get("machine_answer")
                else:
                    human_text, generated_text = item.get("human_text"), item.get("machine_text")
                human_text = clean_text(human_text)
                generated_text = clean_text(generated_text)
                if human_text:
                    records.append({"text": human_text, "label": 0})
                if generated_text:
                    records.append({"text": generated_text, "label": 1})
    for seed in seeds:
        test = seeded_test_records(records, seed, inplace=True)
        write_jsonl(out_root / f"opensrc_platforms_seed{seed}" / "m4.jsonl", test)


def prepare_deepfake(path: Path, out_root: Path, seeds: list[int]) -> None:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            text = clean_text(item.get("text"))
            if text:
                records.append({"text": text, "label": int(item.get("label") != "human")})
    for seed in seeds:
        test = seeded_test_records(records, seed, inplace=True)
        write_jsonl(out_root / f"opensrc_platforms_seed{seed}" / "deepfake.jsonl", test)


def encode_outputs(repo_root: Path, out_root: Path, seeds: list[int], encoder: str, device: str | None, batch_size: int) -> None:
    for seed in seeds:
        raw_dir = out_root / f"opensrc_platforms_seed{seed}"
        emb_dir = out_root / f"opensrc_platforms_seed{seed}_emb"
        for input_path in sorted(raw_dir.glob("*.jsonl")):
            output_path = emb_dir / f"{input_path.stem}_emb.jsonl"
            command = [
                sys.executable,
                "-m",
                "src.encode_posts",
                "--in",
                str(input_path),
                "--out",
                str(output_path),
                "--encoder",
                encoder,
                "--batch_size",
                str(batch_size),
            ]
            if device:
                command.extend(["--device", device])
            subprocess.run(command, cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--multisocial-csv", type=Path, default=None)
    parser.add_argument("--fox8-ndjson", type=Path, default=None)
    parser.add_argument("--tweepfake-splits", type=Path, default=None)
    parser.add_argument("--m4-dir", type=Path, default=None)
    parser.add_argument("--deepfake-jsonl", type=Path, default=None)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    multisocial_csv = args.multisocial_csv or source_root / "multisocial_anonymized.csv"
    fox8_ndjson = args.fox8_ndjson or source_root / "fox8_23_dataset.ndjson"
    tweepfake_splits = args.tweepfake_splits or source_root / "tweepfake_deepfake_text_detection" / "data" / "splits"
    m4_dir = args.m4_dir or source_root / "M4"
    deepfake_jsonl = args.deepfake_jsonl or source_root / "synthetic-text-datasets" / "RedditBot.jsonl"

    prepare_aigtbench(out_root, args.seeds)
    prepare_multisocial(multisocial_csv, out_root, args.seeds)
    prepare_fox8(fox8_ndjson, out_root, args.seeds)
    prepare_tweepfake(tweepfake_splits, out_root, args.seeds)
    prepare_m4(m4_dir, out_root, args.seeds)
    prepare_deepfake(deepfake_jsonl, out_root, args.seeds)
    if args.encode:
        repo_root = Path(__file__).resolve().parents[1]
        encode_outputs(repo_root, out_root, args.seeds, args.encoder, args.device, args.batch_size)
    print(f"Prepared {len(args.seeds)} seed directories under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
