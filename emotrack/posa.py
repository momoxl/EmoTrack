"""Position-Shuffled Augmentation (PoSA).

Each logical example contains two controlled views of the same acoustic target:
the teacher view places it last, while the student view places it at a sampled
non-final position. The fillers, their relative order, answer options, and gold
label are identical across the pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

LETTERS = ("A", "B", "C", "D")
DEFAULT_LABELS = ("anger", "disgust", "fear", "happy", "neutral", "sad", "surprise")

_LABEL_ALIASES = {
    "ang": "anger",
    "anger": "anger",
    "angry": "anger",
    "dis": "disgust",
    "disgust": "disgust",
    "fea": "fear",
    "fear": "fear",
    "exc": "happy",
    "hap": "happy",
    "happy": "happy",
    "joy": "happy",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "sur": "surprise",
    "surp": "surprise",
    "surprise": "surprise",
    "surprised": "surprise",
}
_UTTERANCE_RE = re.compile(r"^(?P<dialogue>.+)_[FM]\d+$")


def normalize_label(label: str) -> str | None:
    """Map common SER label spellings to the canonical seven-class names."""
    return _LABEL_ALIASES.get(label.strip().lower())


def infer_dialogue_id(audio_path: str) -> str:
    """Infer a conservative dialogue id when the manifest does not provide one."""
    stem = Path(audio_path).stem
    match = _UTTERANCE_RE.match(stem)
    return match.group("dialogue") if match else stem


def _resolve_audio_path(raw_path: str, manifest: Path, audio_root: Path | None) -> str:
    path = Path(raw_path.replace("\\", "/"))
    if path.is_absolute():
        return str(path)
    base = audio_root if audio_root is not None else manifest.parent
    return str((base / path).resolve())


def load_manifest(
    manifest_path: str | Path,
    labels: Sequence[str] = DEFAULT_LABELS,
    audio_root: str | Path | None = None,
) -> list[dict]:
    """Load a CSV manifest with path, emotion, and optional dialogue_id columns."""
    manifest = Path(manifest_path)
    allowed = set(labels)
    root = Path(audio_root) if audio_root is not None else None
    rows: list[dict] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_number, raw in enumerate(reader, start=2):
            raw_path = raw.get("path") or raw.get("audio_path") or raw.get("wav_path")
            raw_label = raw.get("emotion") or raw.get("label")
            if not raw_path or not raw_label:
                raise ValueError(f"missing path or emotion at {manifest}:{line_number}")
            emotion = normalize_label(raw_label)
            if emotion not in allowed:
                continue
            path = _resolve_audio_path(raw_path, manifest, root)
            rows.append(
                {
                    "path": path,
                    "emotion": emotion,
                    "dialogue_id": raw.get("dialogue_id") or infer_dialogue_id(path),
                    "dataset": raw.get("dataset", ""),
                }
            )
    if not rows:
        raise ValueError(f"no rows with requested labels in {manifest}")
    return rows


def build_options(gold: str, labels: Sequence[str], rng: random.Random) -> tuple[list[str], str]:
    """Construct a randomized four-way answer set shared by both views."""
    if gold not in labels:
        raise ValueError(f"gold label {gold!r} is not in the label vocabulary")
    distractors = [label for label in labels if label != gold]
    if len(distractors) < 3:
        raise ValueError("PoSA's four-way objective requires at least four labels")
    options = [gold, *rng.sample(distractors, 3)]
    rng.shuffle(options)
    return options, LETTERS[options.index(gold)]


def _choose_fillers(
    rows: Sequence[dict], target: dict, count: int, rng: random.Random
) -> list[dict]:
    candidates = [
        row
        for row in rows
        if row["path"] != target["path"] and row["dialogue_id"] != target["dialogue_id"]
    ]
    if len(candidates) < count:
        raise ValueError(f"not enough independent fillers for target {target['path']}")
    return rng.sample(candidates, count)


def build_paired_views(
    rows: Sequence[dict],
    labels: Sequence[str] = DEFAULT_LABELS,
    num_samples: int = 1000,
    turns: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Create one target-final and one sampled target-earlier view per sample."""
    if turns < 2:
        raise ValueError("turns must be at least two")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    labels = tuple(labels)
    by_emotion: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["emotion"] in labels:
            by_emotion[row["emotion"]].append(row)
    missing = [label for label in labels if not by_emotion[label]]
    if missing:
        raise ValueError(f"manifest has no target candidates for: {', '.join(missing)}")

    rng = random.Random(seed)
    records: list[dict] = []
    for sample_id in range(num_samples):
        emotion = labels[sample_id % len(labels)]
        target = rng.choice(by_emotion[emotion])
        fillers = _choose_fillers(rows, target, turns - 1, rng)
        options, gold_letter = build_options(emotion, labels, rng)
        student_position = rng.randint(1, turns - 1)

        for perm_id, (role, target_position) in enumerate(
            (("teacher", turns), ("student", student_position))
        ):
            ordered = list(fillers)
            ordered.insert(target_position - 1, target)
            records.append(
                {
                    "sample_id": sample_id,
                    "perm_id": perm_id,
                    "role": role,
                    "turns": turns,
                    "target_pos": target_position,
                    "target_path": target["path"],
                    "target_emotion": emotion,
                    "audio_paths": [row["path"] for row in ordered],
                    "turn_emotions": [row["emotion"] for row in ordered],
                    "options": options,
                    "gold_letter": gold_letter,
                }
            )
    return records


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV containing path and emotion")
    parser.add_argument("--output", required=True, help="destination grouped JSONL")
    parser.add_argument("--audio-root", default=None, help="base directory for relative paths")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    rows = load_manifest(args.manifest, labels=labels, audio_root=args.audio_root)
    records = build_paired_views(
        rows,
        labels=labels,
        num_samples=args.num_samples,
        turns=args.turns,
        seed=args.seed,
    )
    write_jsonl(records, args.output)
    print(f"wrote {len(records)} records ({len(records) // 2} pairs) to {args.output}")


if __name__ == "__main__":
    main()

