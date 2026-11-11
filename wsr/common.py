"""Configuration, study rounds and file layout."""
from __future__ import annotations

import json
from pathlib import Path

# The three evaluation rounds of the study plus the standard-edit set on Muse-30B.
#   readers[target] = reader models for that target; the first reader is the target itself (Self framing).
ROUNDS = {
    "full_dose": {"targets": ["27b", "9b"],
                  "readers": {"27b": ["27b", "9b", "muse30b"], "9b": ["9b", "27b", "muse30b"]},
                  "family_cap": 8, "non_destructive_only": False, "framing_control": True},
    "subtle": {"targets": ["27b", "9b"],
               "readers": {"27b": ["27b", "9b"], "9b": ["9b", "27b"]},
               "family_cap": 10, "non_destructive_only": True, "framing_control": False},
    "muse": {"targets": ["muse30b"],
             "readers": {"muse30b": ["muse30b", "27b"]},
             "family_cap": 10, "non_destructive_only": False, "framing_control": False},
    "muse_standard": {"targets": ["muse30b"],
                      "readers": {"muse30b": ["muse30b", "27b", "9b"]},
                      "family_cap": None, "non_destructive_only": False, "framing_control": False},
}

BASE_RUNS = {"base_s0": 0, "base_s1": 100000, "base_s2": 200000, "base_s3": 300000, "base_s4": 400000}
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)
MAX_TOKENS = 8192


def load_cfg(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("work_dir", "work")
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("reader_files_dir", str(Path(cfg["work_dir"]) / "reader_files"))
    return cfg


class Layout:
    def __init__(self, cfg: dict, rnd: str):
        if rnd not in ROUNDS:
            raise SystemExit(f"unknown round {rnd}; one of {list(ROUNDS)}")
        self.cfg, self.rnd = cfg, rnd
        self.root = Path(cfg["work_dir"]) / rnd

    def pool(self, t):        return self.root / "pool" / f"{t}.jsonl"
    def bank(self, t):        return self.root / "bank" / f"{t}.jsonl"
    def measure(self, t):     return self.root / "measure" / t
    def labels(self, t):      return self.root / "labels" / f"{t}.json"
    def judge_cache(self, t): return self.root / "labels" / f"{t}_judge.json"
    def sessions(self):       return self.root / "sessions"
    def scores(self):         return self.root / "scores"
    def reader_files(self, t): return Path(self.cfg["reader_files_dir"]) / self.rnd / t


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
