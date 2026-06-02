# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

CIA (Communication Inference of Agents)는 LLM 기반 멀티 에이전트 시스템(MAS)의 **통신 토폴로지를 역추론**하는 연구 코드입니다. 학습된 생성형 토폴로지 모델(G-Designer 등)의 추론 출력에서 에이전트 간 통신 구조를 복원합니다.

## 주요 실행 명령어

### Stage 1: 생성 토폴로지 모델 학습 (G-Designer)
```bash
cd CIA/GDesigner-main
python experiments/run_gsm8k.py   # GSM8K
python experiments/run_mmlu.py    # MMLU
python experiments/run_humaneval.py  # HumanEval
python experiments/run_svamp.py   # SVAMP
```
실행 후 `.pt` 체크포인트 파일 생성됨.

### Stage 2: 추론 출력 유도 (Reasoning Output Induction)
```bash
cd CIA/CIA
python reasoning_output_induction.py \
  --model_path <.pt 파일 경로> \
  --dataset_path <데이터셋 경로> \
  --domain gsm8k \           # gsm8k | mmlu | svamp | humaneval
  --agent_names MathSolver \ # CodeWriting(humaneval), MathSolver(gsm8k/svamp), AnalyzeAgent(mmlu)
  --agent_nums 4 \
  --decision_method FinalRefer  # FinalRefer(gsm8k/svamp/mmlu), FinalWriteCode(humaneval)
```
출력: `/{domain}/reasoning_outputs/{i_batch}_{i_record}.json`

### Stage 3: 시맨틱 상관관계 모델링 (Topology Inference)
```bash
cd CIA/CIA
python semantic_correlations_modeling.py \
  --data_path <reasoning_outputs 디렉토리> \
  --domain gsm8k \
  --text_encoder /data/llm/all-MiniLM-L6-v2 \
  --epochs 50
```

## 아키텍처 구조

```
CIA/
├── CIA/CIA/          ← 핵심 추론 모듈
└── CIA/GDesigner-main/ ← Stage 1 학습 프레임워크 (서브모듈)
```

### 핵심 CIA 모듈 (`CIA/CIA/`)

**3단계 파이프라인:**

1. **`reasoning_output_induction.py`** — G-Designer Graph를 로드하고 각 데이터 레코드마다 MAS를 5회 실행하여 에이전트 추론 출력과 실제 연결 구조를 JSON으로 저장. GCN 체크포인트를 로드해 `eval_arun()`으로 그래프 실행.

2. **`data_processor.py`** (`GraphDataProcessor`) — Stage 2의 JSON 출력을 파싱하여 PyTorch Geometric `Data` 객체 생성. 에이전트 출력 텍스트와 노드 간 시퀀스 유사도로 노드-인덱스 매핑 (`_match_nodes_to_outputs`). 공간(spatial) 및 시간(temporal) 엣지 모두 인접 행렬로 통합.

3. **`trainer_LLM.py`** (`SelfSupervisedTrainer`) — Self-supervised 방식으로 통신 토폴로지 추론 학습. 각 epoch마다 LLM(GPT)에 에이전트 출력을 전달해 Top-3 엣지를 예측받고 이를 Positive 레이블로 활용. 손실 함수: `rec_loss - s_loss + align_loss + 0.1*sup_loss`.

**모델 컴포넌트 (`models_LLM.py`, `utils.py`):**

- `SelfSupervisedModel`: 핵심 신경망. 두 인코더(`encoder_p`, `encoder_s`)가 에이전트 출력 임베딩을 공통(shared)·고유(private) 표현으로 분리. `TCLineEstimator`(InfoNCE 기반)로 mutual information 추정. `link_predictor` Bilinear 레이어로 엣지 점수 계산.
- `SentenceTransformerEncoder`: `all-MiniLM-L6-v2`로 에이전트 출력 텍스트 임베딩.
- `_client`: `utils.py`에 선언된 OpenAI 호환 클라이언트. `BASE_URL`과 `API_KEY`는 `template.env` 참고.

### GDesigner 서브모듈 (`CIA/GDesigner-main/`)

- **`GDesigner/graph/graph.py`** (`Graph`): MAS 실행 엔진. GCN으로 태스크별 에이전트 연결 확률(logit) 계산 → 공간/시간 엣지 샘플링 → 위상 정렬 기반 에이전트 실행.
- **`GDesigner/gnn/gcn.py`**: 노드 특성(에이전트 역할 임베딩 + 태스크 임베딩)을 입력받아 연결 logit 출력.
- **에이전트 종류**: `CodeWriting`, `MathSolver`, `AnalyzeAgent`, `AdversarialAgent` 등 (`GDesigner/agents/`).
- **도메인별 프롬프트**: `GDesigner/prompt/` 하위 각 벤치마크 파일.

## 환경 설정

`CIA/GDesigner-main/template.env`에 `BASE_URL`과 `API_KEY` 설정 필요.  
`CIA/CIA/utils.py`의 `_client = OpenAI(base_url="", api_key="")` 값도 동일하게 설정.

`reasoning_output_induction.py`의 sys.path는 현재 `/data/CIA/GDesigner-main`으로 하드코딩되어 있으므로 실행 환경에 맞게 수정 필요:
```python
sys.path.append('/data/CIA/GDesigner-main')  # 실제 경로로 변경
```

텍스트 인코더 모델 경로 (`/data/llm/all-MiniLM-L6-v2`)도 로컬 환경에 맞게 조정.

## 데이터 흐름

```
[.pt 체크포인트] + [벤치마크 데이터셋]
        ↓ reasoning_output_induction.py
[/{domain}/reasoning_outputs/*.json]  ← 에이전트 출력 + 그래프 구조
        ↓ semantic_correlations_modeling.py (data_processor + trainer_LLM)
[AUC / F1 / Accuracy 평가 결과]
```

각 JSON 파일 구조: `[[R_star_list, {decision: {...}, nodes: [...]}], ...]` 형태로 5회 반복 실행 결과 저장.
