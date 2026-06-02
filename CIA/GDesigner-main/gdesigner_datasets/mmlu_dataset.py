import re
from typing import List, Dict, Any


ANSWER_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
ANSWER_REV = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class MMLUDataset:
    """
    HuggingFace cais/mmlu 데이터셋을 래핑.
    split: 'dev' → validation, 'val' → validation, 'test' → test
    """

    def __init__(self, split: str, data_path: str = None):
        from datasets import load_dataset
        self.split = split
        hf_split = 'validation' if split in ('dev', 'val') else 'test'
        self._data = load_dataset('cais/mmlu', 'all', split=hf_split)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def record_to_input(self, record: Dict) -> Dict[str, str]:
        """train_mmlu / evaluate_mmlu 용."""
        return {'task': self._format_question(record)}

    def record_to_input_adversial(self, record: Dict) -> Dict[str, str]:
        """reasoning_output_induction.py 용."""
        return {'task': self._format_question(record), 'answer': ANSWER_MAP[record['answer']]}

    def record_to_target_answer(self, record: Dict) -> str:
        return ANSWER_MAP[record['answer']]

    def postprocess_answer(self, raw_answer) -> str:
        """LLM 응답에서 A/B/C/D 추출."""
        text = raw_answer[0] if isinstance(raw_answer, list) else str(raw_answer)
        match = re.search(r'\b([A-D])\b', text)
        return match.group(1) if match else ''

    def _format_question(self, record: Dict) -> str:
        choices = record['choices']
        options = '\n'.join(f"{ANSWER_MAP[i]}. {c}" for i, c in enumerate(choices))
        return f"{record['question']}\n{options}"
