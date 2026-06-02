"""
실험에 필요한 데이터셋을 GDesigner-main/datasets/ 아래에 준비합니다.
- GSM8K  : HuggingFace datasets 라이브러리로 자동 다운로드
- HumanEval : HuggingFace datasets 라이브러리로 자동 다운로드
- SVAMP  : HuggingFace datasets 라이브러리로 자동 다운로드

실행:
    python prepare_datasets.py
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets"


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  저장 완료: {path}  ({len(records)}건)")


def save_json(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {path}  ({len(records)}건)")


# ── GSM8K ──────────────────────────────────────────────────────────────────
def prepare_gsm8k():
    print("\n[1/3] GSM8K 다운로드 중...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    records = []
    for row in ds:
        # answer 필드: "... #### 숫자" 형태 → 숫자만 추출
        raw_answer = row["answer"]
        answer_num = raw_answer.split("####")[-1].strip()
        records.append({
            "question": row["question"],
            "answer": answer_num,
        })
    save_jsonl(records, DATASETS_DIR / "gsm8k" / "gsm8k.jsonl")


# ── HumanEval ──────────────────────────────────────────────────────────────
def prepare_humaneval():
    print("\n[2/3] HumanEval 다운로드 중...")
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    records = []
    for row in ds:
        records.append({
            "task_id":           row["task_id"],
            "prompt":            row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test":              row["test"],
            "entry_point":       row["entry_point"],
        })
    save_jsonl(records, DATASETS_DIR / "humaneval" / "humaneval-py.jsonl")


# ── SVAMP ──────────────────────────────────────────────────────────────────
def prepare_svamp():
    print("\n[3/3] SVAMP 다운로드 중...")
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split="test")
    records = []
    for row in ds:
        records.append({
            "ID":       row["ID"],
            "Body":     row["Body"],
            "Question": row["Question"],
            "Equation": row["Equation"],
            "Answer":   row["Answer"],
            "Type":     row["Type"],
        })
    # run_svamp.py (Stage 1): JSONLReader로 읽음 → jsonl
    save_jsonl(records, DATASETS_DIR / "svamp" / "svamp.jsonl")
    # reasoning_output_induction.py (Stage 2): json.load로 읽음 → json 배열
    save_json(records, DATASETS_DIR / "svamp" / "svamp.json")


if __name__ == "__main__":
    prepare_gsm8k()
    prepare_humaneval()
    prepare_svamp()
    print("\n모든 데이터셋 준비 완료.")
