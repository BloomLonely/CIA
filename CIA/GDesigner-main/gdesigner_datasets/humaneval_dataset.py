from typing import List, Dict


def humaneval_data_process(raw_data: List[Dict]) -> List[Dict]:
    """
    JSONLReader로 읽은 HumanEval 레코드를 실험 스크립트가 기대하는 형식으로 변환.
    run_humaneval.py: record["prompt"], record["test"], record["entry_point"]
    reasoning_output_induction.py: record["task"], record["answer"]
    """
    processed = []
    for i, record in enumerate(raw_data):
        processed.append({
            'task':           record.get('task_id', ''),
            'prompt':         record.get('prompt', ''),
            'test':           record.get('test', ''),
            'entry_point':    record.get('entry_point', ''),
            'answer':         record.get('canonical_solution', ''),
            'step':           i,
        })
    return processed
