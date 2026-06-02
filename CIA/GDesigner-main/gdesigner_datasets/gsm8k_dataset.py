import re
import json
from pathlib import Path
from typing import List, Dict, Any


def gsm_get_predict(text: str) -> str:
    """LLM 응답 텍스트에서 최종 숫자 답을 추출."""
    if not text:
        return ""
    # #### 숫자 패턴 (GSM8K 공식 포맷)
    match = re.search(r'####\s*([\d,\.\-]+)', str(text))
    if match:
        return match.group(1).replace(',', '').strip()
    # 마지막에 등장하는 숫자
    numbers = re.findall(r'[\-]?\d+(?:\.\d+)?', str(text).replace(',', ''))
    return numbers[-1] if numbers else ""


def gsm_data_process(raw_data: List[Dict]) -> List[Dict]:
    """
    JSONLReader로 읽은 GSM8K 레코드를 실험 스크립트가 기대하는
    {task, answer, step} 형식으로 변환.
    """
    processed = []
    for i, record in enumerate(raw_data):
        question = record.get('question', record.get('task', ''))
        answer_raw = record.get('answer', '')
        # "#### 18" 형태에서 숫자만 추출
        answer = gsm_get_predict(answer_raw) if '####' in str(answer_raw) else str(answer_raw)
        processed.append({
            'task': question,
            'answer': answer,
            'step': i,
        })
    return processed


def gsm_data_process_adversial(raw_data: List[Dict]) -> List[Dict]:
    """reasoning_output_induction.py용 변환 (gsm_data_process와 동일 포맷)."""
    return gsm_data_process(raw_data)


def svamp_data_process_adversial(raw_data: List[Dict]) -> List[Dict]:
    """
    SVAMP JSON 배열을 {task, answer, step} 형식으로 변환.
    SVAMP 필드: Body, Question, Answer
    """
    processed = []
    for i, record in enumerate(raw_data):
        body = record.get('Body', '')
        question = record.get('Question', '')
        task = f"{body} {question}".strip()
        answer = str(record.get('Answer', ''))
        processed.append({
            'task': task,
            'answer': answer,
            'step': i,
        })
    return processed
