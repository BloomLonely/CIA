import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy
from pathlib import Path

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.utils.const import GDesigner_ROOT

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            llm_name:str = "",
            domain:str = "",
            mode:str = "FullConnected",
            current_time:str = "",
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    from tqdm import tqdm
    optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=lr)
    graph.gcn.train()
    pbar = tqdm(range(num_iters), desc="[MMLU] Iter", unit="iter")
    for i_iter in pbar:
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        
        for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss)
    
        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()

        avg_util = sum(utilities) / len(utilities) if utilities else 0.0
        pbar.set_postfix({
            "acc": f"{avg_util:.2f}",
            "loss": f"{total_loss.item():.3f}",
            "time": f"{time.time()-start_ts:.1f}s",
            "tokens": f"{int(PromptTokens.instance().value+CompletionTokens.instance().value)}",
        })

    result_dir = Path(f"{GDesigner_ROOT}/result/mmlu")
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    llm_safe = llm_name.replace("/", "_")
    checkpoint_path = checkpoint_dir / f"gcn_{domain}_{llm_safe}_{current_time}_{mode}.pt"
    torch.save({
        'gcn_state_dict': graph.gcn.state_dict(),
        'mlp_state_dict': graph.mlp.state_dict() if hasattr(graph, 'mlp') else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'current_time': current_time,
    }, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
