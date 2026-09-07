#!/usr/bin/env python3
"""Minimal Qwen2-Audio LoRA trainer for PoSA and PoSD."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from emotrack.posd import posa_posd_loss

LETTERS = ("A", "B", "C", "D")
SYSTEM_PROMPT = "You are a helpful assistant."


class GroupedPoSADataset(Dataset):
    """Load and validate target-final/target-earlier PoSA pairs."""

    def __init__(self, jsonl_path: str | Path):
        groups: dict[Any, list[dict]] = defaultdict(list)
        with Path(jsonl_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "sample_id" not in row:
                    raise ValueError(f"missing sample_id at line {line_number}")
                groups[row["sample_id"]].append(row)

        self.groups: list[list[dict]] = []
        for sample_id, group in groups.items():
            teachers = [row for row in group if row.get("role") == "teacher"]
            students = [row for row in group if row.get("role") == "student"]
            if len(teachers) != 1 or not students:
                raise ValueError(f"sample {sample_id} must have one teacher and >=1 student")
            teacher = teachers[0]
            turns = len(teacher["audio_paths"])
            if teacher["target_pos"] != turns:
                raise ValueError(f"sample {sample_id} teacher target is not final")
            for student in students:
                if not 1 <= student["target_pos"] < turns:
                    raise ValueError(f"sample {sample_id} student target is not non-final")
                for field in ("target_path", "options", "gold_letter"):
                    if student[field] != teacher[field]:
                        raise ValueError(f"sample {sample_id} differs in paired field {field}")
                if Counter(student["audio_paths"]) != Counter(teacher["audio_paths"]):
                    raise ValueError(f"sample {sample_id} views do not contain the same audio")
            self.groups.append([teacher, *sorted(students, key=lambda row: row["perm_id"])])
        if not self.groups:
            raise ValueError(f"no PoSA groups found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> list[dict]:
        return self.groups[index]


def build_prompt(record: dict) -> str:
    turns = len(record["audio_paths"])
    position = int(record["target_pos"])
    ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(position, f"{position}th")
    options = "\n".join(
        f"{letter}. {emotion}" for letter, emotion in zip(LETTERS, record["options"])
    )
    return (
        f"You just heard audio clips from a {turns}-turn conversation. "
        "What is the emotion expressed in the speaker's voice during "
        f"the {ordinal} turn?\n\n"
        f"Choose one from the following four options:\n{options}\n\n"
        "Reply with ONLY the letter of your chosen option (A, B, C, or D)."
    )


def build_messages(record: dict) -> list[dict]:
    content = [
        {"type": "audio", "audio_url": path} for path in record["audio_paths"]
    ]
    content.append({"type": "text", "text": build_prompt(record)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def validate_audio_features(encoded: dict[str, Any], expected_audio_count: int) -> None:
    """Fail closed if Qwen2-Audio silently produced a text-only batch."""
    for name in ("input_features", "feature_attention_mask"):
        if name not in encoded or not torch.is_tensor(encoded[name]):
            raise RuntimeError(f"processor did not create required acoustic tensor: {name}")
    if encoded["input_features"].ndim != 3:
        raise RuntimeError("input_features must be rank 3")
    if encoded["input_features"].shape[0] != expected_audio_count:
        raise RuntimeError("number of acoustic features does not match audio placeholders")


class Qwen2AudioPoSD:
    def __init__(self, args: argparse.Namespace):
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[args.dtype]
        self.device = torch.device(args.device)
        self.processor = AutoProcessor.from_pretrained(args.model)
        base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=dtype
        )
        base_model.config.use_cache = False
        base_model.requires_grad_(False)
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$",
        )
        self.model = get_peft_model(base_model, lora).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.letter_ids = torch.tensor(
            [self._single_token_id(letter) for letter in LETTERS],
            dtype=torch.long,
            device=self.device,
        )

    def _single_token_id(self, text: str) -> int:
        ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"answer {text!r} is not represented by one token: {ids}")
        return ids[0]

    def encode(self, record: dict, waveforms: dict[str, Any]) -> dict[str, torch.Tensor]:
        audios = [waveforms[path] for path in record["audio_paths"]]
        rendered = self.processor.apply_chat_template(
            build_messages(record), add_generation_prompt=True, tokenize=False
        )
        encoded = self.processor(
            text=rendered,
            audio=audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        validate_audio_features(encoded, len(audios))
        return {key: value.to(self.device) for key, value in encoded.items()}

    def answer_logits(self, record: dict, waveforms: dict[str, Any]) -> torch.Tensor:
        encoded = self.encode(record, waveforms)
        output = self.model(**encoded, use_cache=False)
        last_index = int(encoded["attention_mask"][0].sum().item()) - 1
        return output.logits[0, last_index, self.letter_ids].float()


def load_waveforms(group: list[dict], sampling_rate: int) -> dict[str, Any]:
    paths = dict.fromkeys(path for row in group for path in row["audio_paths"])
    return {
        path: librosa.load(path, sr=sampling_rate, mono=True)[0] for path in paths
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen2-Audio model path or Hub id")
    parser.add_argument("--train-jsonl", required=True, help="grouped JSONL produced by PoSA")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--posd-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--posd-warmup-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-groups", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_groups < 1:
        raise ValueError("gradient accumulation must be positive")
    if not 0 <= args.posd_warmup_fraction <= 1:
        raise ValueError("posd warmup fraction must be in [0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    dataset = GroupedPoSADataset(args.train_jsonl)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    runner = Qwen2AudioPoSD(args)
    parameters = [parameter for parameter in runner.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    steps_per_epoch = math.ceil(len(dataset) / args.gradient_accumulation_groups)
    planned_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    warmup_steps = max(1, round(planned_steps * args.posd_warmup_fraction))
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    pending_groups = 0

    for epoch in range(args.epochs):
        for group_index, group in enumerate(loader, start=1):
            waveforms = load_waveforms(group, runner.sampling_rate)
            teacher_logits = runner.answer_logits(group[0], waveforms)
            student_logits = [runner.answer_logits(row, waveforms) for row in group[1:]]
            gold_index = LETTERS.index(group[0]["gold_letter"])
            current_weight = args.posd_weight * min(
                1.0, (optimizer_step + 1) / warmup_steps
            )
            losses = posa_posd_loss(
                teacher_logits,
                student_logits,
                gold_index,
                posd_weight=current_weight,
                temperature=args.temperature,
            )
            (losses.total / args.gradient_accumulation_groups).backward()
            pending_groups += 1

            end_of_epoch = group_index == len(loader)
            if pending_groups == args.gradient_accumulation_groups or end_of_epoch:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                pending_groups = 0

                if optimizer_step == 1 or optimizer_step % args.log_every == 0:
                    print(
                        f"epoch={epoch + 1} step={optimizer_step}/{planned_steps} "
                        f"total={losses.total.item():.4f} posa={losses.posa.item():.4f} "
                        f"posd={losses.posd.item():.4f} lambda={current_weight:.3f} "
                        f"gate={int(losses.teacher_correct)}",
                        flush=True,
                    )
                if args.max_steps > 0 and optimizer_step >= args.max_steps:
                    break
        if args.max_steps > 0 and optimizer_step >= args.max_steps:
            break

    adapter_dir = output_dir / "adapter"
    runner.model.save_pretrained(adapter_dir)
    runner.processor.save_pretrained(output_dir / "processor")
    print(f"saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()

