1. Tokenization & Data Pipeline (scripts/tok_train.py, nanochat/tokenizer.py, nanochat/dataloader.py)
    - 학습을 위한 데이터 준비 파이프라인으로 이 단계는 고정된 크기의 행 텐서($B \times T$)를 만들어 모델의 그래디언트를 역전파하기 위한 학습용 배치(Batch)를 찍어내는 공장입니다.
    1. 선행 단계: BPE 어휘 사전 학습 및 빌드 (Tokenizer Training: scripts/tok_train.py, rustbpe)
        - inputs: Parquet 학습 샤드 전체에서 로드되어 최대 10,000자로 잘린 단일 텍스트 스트림 `text_iterator` (누적 문자 수: `max_chars=2,000,000,000`, 약 20억 자)
        - algorithm: 
            1. 기본 어휘 초기화: 256개의 원시 UTF-8 바이트(ID: $0 \sim 255$)를 기본 어휘로 등록합니다.
            2. 정규식 청크 분할: 텍스트 스트림을 `SPLIT_PATTERN` 정규식에 따라 문자열 청크로 분할합니다.
                - 적용 정규식: `SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""`
            3. 청크 내 빈도수 집계 및 반복 병합 (rustbpe): 
                - 각 청크 내부에서만 인접한 바이트 쌍의 등장 빈도를 집계합니다 ((청크 1의 맨 뒷 바이트, 청크 2의 맨 앞 바이트)와 같이 청크 경계를 넘어선 바이트 쌍은 집계에서 제외).
                - 가장 높은 빈도의 바이트 쌍을 새 토큰 ID로 등록하는 과정을 목표 어휘 크기(`vocab_size - len(SPECIAL_TOKENS) = 32768 - 9 = 32759`)까지 반복하여 병합 우선순위 딕셔너리(`mergeable_ranks`)를 빌드합니다.
            4. 특수 토큰 매핑 및 인코더 직렬화:
                - 총 9개의 특수 토큰(`<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`)에 오프셋 ID($32759 \sim 32767$)를 부여하여 최종 `tiktoken.Encoding` 객체로 패키징하고 `tokenizer.pkl`로 저장합니다.
            5. Token Bytes 캐싱: 각 토큰 ID별 원시 바이트 길이를 매핑한 1D 텐서 `token_bytes.pt` (크기: 32768)를 생성 및 저장합니다 (BPB 평가 시 Vocab-Size 불변 손실 계산에 활용).
        - outputs: 디스크에 저장된 `tokenizer.pkl` (BPE 인코딩 룩업 테이블) 및 `token_bytes.pt` (토큰 바이트 크기 매핑 텐서)
    2. 분기 1: Pretraining Data Pipeline (nanochat/dataloader.py)
        1. Parquet Shard Loading (_document_batches)
            - inputs: 디스크에 저장된 사전학습 데이터 샤드 파일들 (`shard_00000.parquet` ~ `shard_06541.parquet`)
            - algorithm: DDP 랭크 및 재개 상태(`resume_state_dict`)에 맞춰 Row Group을 순회하며 원시 텍스트 문서를 128개 단위(`tokenizer_batch_size=128`)로 스트리밍 로드합니다.
            - outputs: 원시 텍스트 문서 배치 `List[str]` (길이: 128)
        2. BOS-Aligned Batch BPE Tokenization (tiktoken)
            - inputs: 원시 텍스트 문서 배치 `List[str]` (길이: 128)
            - algorithm: 
                - 사전 로드된 토크나이저를 통해 멀티스레드(`tokenizer_threads=4`) 병렬 인코딩을 수행합니다.
                - 입력 텍스트의 각 문서를 `SPLIT_PATTERN` 정규식에 따라 독립적인 문자열 청크(`List[List[str]]`)로 분할합니다.
                - 각 청크를 UTF-8 바이트 시퀀스로 분해한 뒤, 학습 단계에서 생성된 `mergeable_ranks`의 우선순위에 따라 청크 내부 바이트들을 탐욕적(Greedy)으로 병합하여 최종 정수 토큰 ID(`List[List[int]]`)로 매핑합니다.(예: `[[15496, 995, 0], [40, 2873, 220, 2026, ...], ...]`)
                - `prepend=bos_token` 옵션에 따라 모든 개별 문서 시퀀스의 맨 앞(`0`번 인덱스)에 `<|bos|>` 특수 토큰 ID(`32759`)를 삽입합니다.
            - outputs: 맨 앞에 `<|bos|>` ID가 포함된 문서별 토큰 ID 리스트 `List[List[int]]` (길이: 128) (예: `[[32759, 15496, 995, ...], [32759, 40, 2873, ...], ...]`)
        3. Document Buffering
            - inputs: 인코딩 완료된 128개 단위의 문서 토큰 리스트 `List[List[int]]`
            - algorithm: 
                - `len(doc_buffer) < buffer_size(1000)`일 때마다 1~2번 단계를 호출(`refill_buffer`)하여 `doc_buffer`에 순차적으로 `append`합니다 (예: 999개일 때 128개가 추가되어 약 1,000~1,127개 수준 유지).
                - 이를 통해 다음 단계의 Best-Fit 패킹 알고리즘이 남은 시퀀스 길이에 꼭 맞는 최적의 문서를 탐색할 수 있는 충분한 후보군 풀(Pool)을 확보합니다.
            - outputs: 인코딩된 문서들이 대기 중인 2차원 리스트 풀 `doc_buffer` (`List[List[int]]`, 문서 개수 `len(doc_buffer) >= 1000` 유지)
        4. Best-Fit Crop Bin Packing
            - inputs: 
                - 인코딩된 문서 후보 풀 `doc_buffer` (`List[List[int]]`)
                - 배치 크기 $B$ (`device_batch_size`), 시퀀스 길이 $T$ (`max_seq_len`)
                - 목표 행 용량 `row_capacity = T + 1` (Next-token 타깃 슬라이싱을 위한 $+1$ 오프셋 포함)
            - algorithm:
                - 행 버퍼 초기화: $B$ 개의 행을 구성하기 위해 크기가 $B \times (T + 1)$ 인 `row_buffer` 텐서를 생성하고 각 행(인덱스 `row_idx = 0 ... B-1`)을 독립적으로 순차 패킹합니다.
                - 탐욕적 Best-Fit 문서 적재 (Greedy Best-Fit Search):
                    1. 현재 행의 남은 토큰 용량(`remaining = row_capacity - pos`)을 계산합니다.
                    2. `doc_buffer` 내의 전체 문서 중 `len(doc) <= remaining` 조건을 만족하면서 길이가 가장 긴(`best_len`) 최적의 문서를 탐색(`best_idx`)합니다.
                    3. 적합한 문서를 발견하면 `doc_buffer`에서 해당 문서를 추출(`pop(best_idx)`)하여 `row_buffer[row_idx, pos : pos + doc_len]`에 배치하고, 현재 쓰기 위치를 갱신합니다 (`pos += doc_len`).
                    4. 남은 용량이 부족하여 버퍼 내 어떤 문서도 들어갈 수 없을 때까지 1~3 과정을 반복합니다.
                - 최소 손실 강제 자르기 (Shortest-Doc Crop):
                    1. 버퍼 내 어떠한 문서도 남은 용량 `remaining`에 온전히 들어가지 못하면, `doc_buffer`에서 길이가 가장 짧은 문서(`min(doc_buffer, key=len)`)를 탐색하여 추출(`pop(shortest_idx)`)합니다 (전체 문서 중 잘려 버려지는 토큰 비율을 최소화하기 위함).
                    2. 해당 문서의 앞부분 토큰을 남은 공간 크기만큼 자른 슬라이스(`doc[:remaining]`)를 `row_buffer[row_idx, pos : pos + remaining]`에 채워 넣어 행을 정확히 $T + 1$ 길이로 가득 채웁니다.
                    3. 잘려 나가고 남은 뒷부분 토큰은 재사용되지 않고 완전히 폐기됩니다 (nanochat/dataloader.py에 따르면 자르기로 인해 전체 토큰의 약 $35\%$ 가 손실되나, 모든 행의 첫 토큰이 항상 `<|bos|>`로 시작하여 완벽한 인과적 어텐션 문맥을 보존하며 패딩 토큰 낭비 $0\%$ 를 달성).
            - outputs: 패딩 없이 유효 토큰으로 $T + 1$ 길이만큼 꽉 채워진 $B \times (T + 1)$ 크기의 텐서 `row_buffer`
        5. Autoregressive Shifted Slicing (Inputs $x_{0:T-1}$, Targets $x_{1:T}$)
            - inputs: 
                - 이전 단계에서 패킹 완료된 행 버퍼 `row_buffer` (텐서 형태: `Tensor[int64]`, 크기: $B \times (T + 1)$)
                - 사전 할당된 고정 메모리 CPU 버퍼 `cpu_buffer` (크기: $2 \times B \times T$) 및 이에 대한 슬라이스 뷰인 `cpu_inputs`, `cpu_targets` (각각 크기: $B \times T$)
            - algorithm:
                - 자기회귀 시프트 슬라이싱 (Shifted Slicing):
                    1. `row_buffer`의 각 행에서 마지막 토큰을 제외한 앞부분 $T$ 개의 토큰(`row_buffer[:, :-1]`)을 슬라이싱하여 모델의 순방향 입력 시퀀스 $x$ 로 할당합니다.
                    2. `row_buffer`의 각 행에서 첫 번째 토큰을 제외한 뒷부분 $T$ 개의 토큰(`row_buffer[:, 1:]`)을 슬라이싱하여 각 입력 토큰 위치에서 맞춰야 할 정답(Next-Token) 타깃 시퀀스 $y$ 로 할당합니다.
                - 사전 할당된 CPU 고정 메모리(Pinned Memory) 뷰에 직접 복사:
                    - 매 스텝마다 새로운 텐서를 메모리에 할당하는 오버헤드(Garbage Collection 및 VRAM 단편화)를 제거하기 위해, 사전 할당된 `cpu_buffer`의 앞/뒤 뷰에 `.copy_()` 연산으로 직접 덮어씁니다.
                    - `cpu_inputs.copy_(row_buffer[:, :-1])`
                    - `cpu_targets.copy_(row_buffer[:, 1:])`
                - `cpu_buffer` 메모리 레이아웃:
                    - 전체 $2 \times B \times T$ 개의 연속된 1차원 평탄화(Flat) 메모리 구조를 가집니다.
                    - 앞쪽 절반 $[0 : B \times T]$ 구간은 `cpu_inputs` ($B \times T$), 뒤쪽 절반 $[B \times T : 2 \times B \times T]$ 구간은 `cpu_targets` ($B \times T$)가 차지하여, 다음 단계에서 GPU로 단 1회의 연속 메모리 복사(Single HtoD Transfer)가 가능하도록 정렬됩니다.
            - outputs: `cpu_buffer`에 입력과 타깃으로 정렬 복사된 CPU 텐서 뷰 `cpu_inputs` (크기: $B \times T$), `cpu_targets` (크기: $B \times T$)
        6. GPU Host-to-Device Transfer
            - inputs: 
                - `cpu_inputs`, `cpu_targets`가 앞뒤로 연속 채워진 고정 메모리 버퍼 `cpu_buffer` (텐서 형태: `Tensor[int64]`, 크기: $2 \times B \times T$, `pin_memory=True`)
                - 디바이스 측에 사전 할당된 영구 텐서 버퍼 `gpu_buffer` (크기: $2 \times B \times T$, 위치: `device`) 및 이에 대한 $2\text{D}$ 슬라이스 뷰 `inputs`, `targets` (각각 크기: $B \times T$)
                - 데이터로더 샤드 순회 상태 메타데이터 딕셔너리 `state_dict` (`{"pq_idx": int, "rg_idx": int, "epoch": int}`)
            - algorithm:
                - 단일 호출 비동기 HtoD 메모리 복사 (Single Host-to-Device Transfer via DMA):
                    - CUDA 환경일 경우 `gpu_buffer.copy_(cpu_buffer, non_blocking=True)`를 실행하여, CPU 고정 메모리(Pinned Memory)로부터 GPU VRAM으로 $2 \times B \times T$ 개의 토큰 텐서 덩어리를 PCIe 버스를 통해 단 1회의 비동기 DMA(Direct Memory Access) 전송으로 일괄 복사합니다.
                    - `non_blocking=True` 옵션을 통해 호스트(CPU)는 GPU 복사 완료를 동기적으로 기다리지 않고 즉시 제어권을 반환하므로, GPU가 연산을 수행하는 동안 다음 배치를 준비하는 파이프라이닝(Data Prefetching)이 가능합니다.
                - 영구 텐서 뷰(View) 공유:
                    - `gpu_buffer`의 앞쪽 절반 뷰인 `inputs` ($gpu\_buffer[:B \times T].\text{view}(B, T)$)와 뒤쪽 절반 뷰인 `targets` ($gpu\_buffer[B \times T:].\text{view}(B, T)$)는 사전 할당된 상태를 그대로 공유하므로, 매 스텝마다 새로운 GPU 메모리를 할당하거나 해제하지 않고 데이터만 갱신됩니다.
                - 제너레이터 반환 (`yield`):
                    - 최종 준비된 `inputs`, `targets`와 재개용 체크포인트 저장을 위한 데이터로더 상태 `state_dict`를 학습 루프로 전달합니다 (`yield inputs, targets, state_dict`).
            - outputs: GPU VRAM 상에 준비된 순방향 입력 텐서 `inputs` (크기: $B \times T$, 타입: `torch.long`), 정답 타깃 텐서 `targets` (크기: $B \times T$, 타입: `torch.long`), 재개 상태 딕셔너리 `state_dict`
    3. 분기 2: SFT / Chat Data Pipeline (scripts/chat_sft.py, tasks/common.py, nanochat/tokenizer.py)
        1. Task Mixture Loading & Deterministic Shuffling (TaskMixture, tasks/common.py)
            - inputs: 
                - 개별 SFT 태스크 데이터셋 인스턴스 목록:
                    - `SmolTalk(split="train")`: 일반 대화 능력 학습용 데이터 (약 46만 행)
                    - `MMLU(subset="all", split="auxiliary_train")`: 객관식 지식 질의응답 학습용 데이터 (약 10만 행, `--mmlu-epochs=3` 옵션에 따라 3회 반복 오버샘플링)
                    - `GSM8K(subset="main", split="train")`: 초등 수학 및 계산기 도구 호출 학습용 데이터 (약 8천 행, `--gsm8k-epochs=4` 옵션에 따라 4회 반복 오버샘플링)
                    - 전달 리스트 형태: `train_tasks = [SmolTalk, MMLU, MMLU, MMLU, GSM8K, GSM8K, GSM8K, GSM8K]` (총 8개 인스턴스)
                - 고정 난수 시드 (`seed=42`)
            - algorithm:
                - HuggingFace Parquet 자동 다운로드 및 캐싱 (`load_hub_dataset`):
                    - 각 태스크 데이터셋 초기화 시 Hub API로부터 해당 서브셋/스플릿의 Parquet 샤드 목록을 쿼리하여 로컬 캐시 디렉터리(`~/.cache/nanochat/task_data/`)에 다운로드합니다.
                    - 분산 환경(DDP)에서는 `FileLock`을 통해 랭크 0 프로세스만 다운로드를 실행하고 나머지 랭크는 대기 후 캐시된 파일을 `pyarrow.concat_tables`로 병합하여 로드합니다.
                - 가상 데이터셋 오버샘플링 (Oversampling via Duplication):
                    - 특정 능력을 강화하기 위해 데이터셋 리스트에 동일한 태스크 인스턴스를 중복 전달하여 물리적 데이터 복제 없이 가상 길이만 배수만큼 확장합니다 (`lengths = [len(task) for task in tasks]`).
                    - 예: 원본 8,000건인 GSM8K를 4번 등록하여 전체 유효 행 수를 $8,000 \times 4 = 32,000$ 건으로 확장.
                - 2차원 평탄화 인덱스 매핑 (Flattened Index Mapping):
                    - 전체 결합 대화 개수 $N_{\text{total}} = \sum \text{len}(\text{task})$ (약 80만 건)를 계산하고, 각 개별 완결 대화(문서)의 고유 주소를 가리키는 `(태스크_인덱스, 대화_행_인덱스)` 순서쌍 리스트를 생성합니다.
                    - 예: `index_map = [(0, 0), (0, 1), ..., (1, 0), ..., (4, 0), ...]` (여기서 `(4, 12)`는 4번 태스크(GSM8K)의 12번째 수학 문제 전문 1건을 의미)
                - 결정론적 인터리빙 셔플링 (Deterministic Interleaved Shuffling):
                    - 파국적 망각(Catastrophic Forgetting)을 방지하기 위해 `random.Random(42).shuffle(self.index_map)`을 적용하여 태스크 종류가 학습 전체에 골고루 깍지 끼듯 교차(Interleaving)되도록 단일 순서열로 재배치합니다.
                    - 예: 셔플링 후 `index_map = [(0, 1542), (4, 12), (1, 8940), (0, 31102), (7, 401), ...]` (일반 대화 $\to$ 수학 $\to$ 객관식 $\to$ 일반 대화 순으로 균형 있게 섞임)
                - DDP 분산 스트라이딩 접근:
                    - 각 분산 랭크는 고유 랭크 오프셋부터 시작하여 월드 크기 간격으로 인덱스를 건너뛰며 순회(`cursor += ddp_world_size`)함으로써 프로세스 간 데이터 중복 없이 독립적인 대화를 순차 스트리밍합니다.
                    - 예: 8개 GPU 환경(`ddp_world_size=8`)일 때 Rank 0은 `index_map[0, 8, 16, ...]`, Rank 1은 `index_map[1, 9, 17, ...]`을 읽음.
            - outputs: 단일 통합 인터페이스로 래핑되어 인터리빙 순서로 완결된 대화 딕셔너리(`Conversation`)를 반환하는 `TaskMixture` 인스턴스 `train_dataset` (약 80만 행 규모)
                - 예: `train_dataset[cursor]` 호출 시 반환되는 대화 객체 예시:
                    - 일반 대화 (SmolTalk): `{"messages": [{"role": "user", "content": "안녕?"}, {"role": "assistant", "content": "안녕하세요!"}]}`
                    - 도구 사용 대화 (GSM8K): `{"messages": [{"role": "user", "content": "사과가 5개..."}, {"role": "assistant", "content": [{"type": "text", "text": "계산: "}, {"type": "python", "text": "5+3"}, {"type": "python_output", "text": "8"}, {"type": "text", "text": "총 8개"}]}]}`
        2. System Message Merging (render_conversation, nanochat/tokenizer.py)
            - inputs: 
                - 이전 단계(`train_dataset`)에서 인출된 단일 대화 딕셔너리 `conversation` (예: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}`)
            - algorithm:
                - 시스템 프롬프트 탐지 및 딥카피 (Deepcopy on System Message):
                    - 대화의 첫 번째 메시지 역할이 시스템 메시지(`conversation["messages"][0]["role"] == "system"`)인지 검사합니다.
                    - 시스템 메시지가 존재할 경우, 원본 데이터셋 객체의 변형(In-place Mutation)을 방지하기 위해 `copy.deepcopy(conversation)`로 독립적인 대화 복사본을 생성합니다.
                - 유저 메시지로의 인라인 병합 (System-to-User Inlining):
                    - 시스템 메시지 직후의 두 번째 메시지가 유저 메시지(`messages[1]["role"] == "user"`)인지 확인합니다.
                    - 시스템 메시지 내용과 유저 메시지 내용을 두 개의 개행 문자(`\n\n`)로 연결하여 유저 메시지 본문에 덮어씁니다 (`messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]`).
                    - 앞자리를 차지하던 0번 시스템 메시지를 리스트에서 슬라이싱 제거(`messages = messages[1:]`)하여, 대화의 시작점이 항상 `user` 역할이 되도록 표준화합니다.
                - 구조 단언 검증 (Sanity Assertion):
                    - 병합 후 남은 메시지 시퀀스의 길이가 최소 1개 이상(`len(messages) >= 1`)이며, 대화가 반드시 사용자의 입력부터 시작하도록 엄격히 보장합니다.
            - outputs: 시스템 메시지가 첫 번째 유저 메시지에 안전하게 흡수 통합되어 `user` $\to$ `assistant` 교차 턴 구조로 정제된 대화 딕셔너리 `conversation`
                - 예: 
                    - 입력: `{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}]}`
                    - 출력: `{"messages": [{"role": "user", "content": "You are a helpful assistant.\n\nHello"}]}`
        3. Supervised Loss Masking & Chat Tokenization (render_conversation, nanochat/tokenizer.py)
            - inputs: 
                - 이전 단계에서 시스템 메시지가 병합된 대화 딕셔너리 `conversation`
                - 학습 허용 최대 토큰 제한 길이 `max_tokens=2048`
                - 사전 빌드된 토크나이저 인코더 `self.enc` 및 특수 토큰 매핑 룩업 테이블
            - algorithm:
                - 특수 토큰 식별자 획득:
                    - 문서 경계 및 대화 제어 특수 토큰 ID들을 캐시로부터 로드합니다: `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`.
                - 대화 시퀀스 토큰화 및 지도 손실 마스킹 (Loss Masking):
                    1. 문서 시작 토큰 `<|bos|>`를 가장 앞에 삽입하며, 손실 마스크는 $0$ 으로 설정합니다 (`add_tokens(bos, 0)`).
                    2. 메시지 리스트(`messages`)를 순회하며 `user`와 `assistant` 역할의 엄격한 교차 등장 여부를 검증합니다 (`must_be_from = "user" if i % 2 == 0 else "assistant"`).
                    3. `user` 메시지 처리:
                        - `<|user_start|>` 태그를 삽입하고, 유저 텍스트 본문을 BPE 인코딩한 토큰 시퀀스를 붙인 뒤 `<|user_end|>` 태그로 닫습니다.
                        - 유저는 모델이 생성할 대상이 아니므로, 포함된 모든 토큰의 손실 마스크는 $0$ 으로 부여합니다.
                    4. `assistant` 메시지 처리:
                        - 시작 태그 `<|assistant_start|>`를 삽입하고, 시작을 유도하는 프롬프트 역할이므로 마스크는 $0$ 을 부여합니다.
                        - 단순 텍스트(`str`)인 경우: 답변 텍스트를 BPE 인코딩하여 추가하고, 모델이 학습해야 할 정답 대상이므로 손실 마스크 $1$ 을 부여합니다.
                        - 멀티파트 도구 사용(`list` of parts, GSM8K 등)인 경우:
                            - `part["type"] == "text"`: 텍스트 토큰을 추가하고 손실 마스크 $1$ 을 부여합니다.
                            - `part["type"] == "python"`: 어시스턴트가 직접 작성해야 하는 코드이므로, `<|python_start|>`와 `<|python_end|>` 태그 및 내부 파이썬 코드 토큰 전체에 손실 마스크 $1$ 을 부여합니다.
                            - `part["type"] == "python_output"`: 파이썬 인터프리터 샌드박스가 반환한 외부 실행 결과이므로, 모델이 생성하는 것이 아닙니다. 따라서 `<|output_start|>`, 결과 텍스트 토큰, `<|output_end|>` 전체에 손실 마스크 $0$ 을 부여합니다.
                        - 답변 종료 태그 `<|assistant_end|>`를 닫는 토큰으로 삽입하며, 모델이 스스로 생성을 멈추는 법(EOS)을 배워야 하므로 손실 마스크 $1$ 을 부여합니다.
                - 시퀀스 길이 강제 절단 (Hard Truncation):
                    - OOM(Out of Memory) 방지를 위해 토큰 ID 리스트와 마스크 리스트를 모두 최대 길이 제한선으로 슬라이싱합니다: `ids = ids[:max_tokens]`, `mask = mask[:max_tokens]`.
            - outputs: 
                - 대화 전체가 단일 1D 정수 ID 시퀀스로 인코딩된 토큰 리스트 `ids` (`List[int]`)
                - 각 토큰 위치의 지도 학습 손실 반영 여부를 나타내는 이진 마스크 리스트 `mask` (`List[int]`, 값은 $0$ 또는 $1$, 길이는 `len(ids)`와 동일)
                - 예: 
                    - `ids`:  `[<|bos|>, <|user_start|>, '1+1은?', <|user_end|>, <|assistant_start|>, '2입니다.', <|assistant_end|>]`
                    - `mask`: `[   0   ,       0       ,    0    ,      0     ,         0          ,     1    ,        1        ]`
        4. Conversation Buffering (conv_buffer, scripts/chat_sft.py)
            - inputs: 
                - 이전 단계에서 토큰화 및 마스킹된 대화 쌍 `(ids, mask)` (`Tuple[List[int], List[int]]`)
                - 셔플된 통합 데이터셋 인스턴스 `train_dataset`
                - 버퍼 유지 목표 크기 `buffer_size=100`
                - 분산 학습 랭크 식별자 (`ddp_rank`, `ddp_world_size`)
            - algorithm:
                - 버퍼 잔여 용량 모니터링 및 트리거:
                    - 현재 대화 버퍼 풀의 크기가 임계치보다 작은지 검사합니다 (`len(conv_buffer) < buffer_size`).
                    - 버퍼 내 남은 대화 수가 부족할 때마다 `refill_buffer()` 보충 서브루틴을 호출하여 정확히 `buffer_size(100)` 개를 다시 채울 때까지 순차적으로 데이터를 당겨옵니다.
                - DDP 독립 커서 전진 및 렌더링 파이프라인 호출:
                    1. 각 GPU 랭크는 자신의 고유 커서 위치에서 대화 데이터를 꺼냅니다 (`conversation = dataset[cursor]`).
                    2. 1.3.2 및 1.3.3 단계를 거쳐 시스템 메시지가 통합되고 토큰화/마스킹이 완료된 `(ids, mask)` 튜플을 획득합니다 (`ids, mask = tokenizer.render_conversation(conversation)`).
                    3. 처리된 튜플을 `conv_buffer` 리스트에 추가합니다 (`conv_buffer.append((ids, mask))`).
                    4. 다음 대화를 위해 커서를 전체 랭크 수만큼 전진시킵니다 (`cursor += ddp_world_size`).
                - 에포크 순환 제어:
                    - 탐색 커서가 전체 데이터셋 크기를 초과할 경우(`cursor >= dataset_size`), 모듈러 연산으로 인덱스를 리셋하고 에포크를 증가시킵니다 (`cursor = cursor % dataset_size`, `epoch += 1`).
                - 후보 풀의 역할:
                    - 다음 단계인 Best-Fit 패킹 알고리즘이 빈 행 공간에 온전히 들어갈 수 있는 최적 길이의 대화를 고를 수 있도록, 다양한 길이 분포를 가진 100개의 대화 후보군 풀을 항시 유지합니다.
            - outputs: 인코딩 및 마스킹된 대화 튜플들이 대기 중인 풀 `conv_buffer` (`List[Tuple[List[int], List[int]]]`, `len(conv_buffer) >= 100` 유지)
        5. Best-Fit Pad Bin Packing (Zero-Loss Padding, scripts/chat_sft.py)
            - inputs: 
                - 인코딩 및 마스킹된 대화 풀 `conv_buffer` (`List[Tuple[List[int], List[int]]]`)
                - 디바이스 배치 크기 $B$ (`device_batch_size`), 시퀀스 길이 $T$ (`max_seq_len`)
                - 목표 행 용량 `row_capacity = T + 1` (Next-token 예측 시프트를 위한 $+1$ 오프셋 포함)
                - 패딩 대체용 특수 토큰 `<|bos|>` ID (`bos_token`)
            - algorithm:
                - 행 버퍼 수집기 초기화:
                    - 한 배치를 구성할 `rows` (토큰 ID 목록), `mask_rows` (마스크 목록), `row_lengths` (패딩 제외 실제 유효 콘텐츠 길이 목록) 리스트를 초기화합니다.
                - 배치 행 순회 루프 (총 $B$ 개의 행 생성):
                    - 각 행마다 빈 리스트 `row = []`, `mask_row = []`, 패딩 여부 플래그 `padded = False`를 두고 `len(row) < row_capacity`를 만족할 때까지 채웁니다.
                    1. 버퍼 수량 보장: `len(conv_buffer) < buffer_size`이면 `refill_buffer()`를 호출하여 후보 풀을 보충합니다.
                    2. 잔여 공간 계산: 현재 행의 남은 토큰 용량 `remaining = row_capacity - len(row)`를 계산합니다.
                    3. 최적 대화 탐색 (Best-Fit Search):
                        - `conv_buffer` 내의 대화 중 `len(conv) <= remaining` 조건을 만족하면서 길이가 가장 긴(`best_len`) 대화 인스턴스(`best_idx`)를 탐색합니다.
                    4. 완전 수용 분기 (`best_idx >= 0`):
                        - 조건에 맞는 대화를 찾으면 버퍼에서 추출(`pop(best_idx)`)하여 `row` 및 `mask_row`에 온전히 이어 붙입니다 (`row.extend(conv)`, `mask_row.extend(conv_mask)`).
                        - 실제 소비량 카운터를 전진시킵니다 (`consumed += ddp_world_size`).
                    5. 공간 부족 및 비손실 패딩 분기 (`best_idx == -1`):
                        - 남은 공간 `remaining`에 온전히 들어갈 수 있는 대화가 버퍼에 단 하나도 없으면, **절대 대화를 자르지 않습니다(No Cropping)**.
                        - 현재까지 들어간 유효 토큰 수(`content_len = len(row)`)를 기록합니다.
                        - 남은 공간 전체를 `<|bos|>` 토큰으로 패딩 채움하고, 대응하는 마스크는 $0$ 으로 채운 뒤 행 패킹을 조기 마감합니다 (`row.extend([bos_token] * remaining)`, `mask_row.extend([0] * remaining)`, `padded = True`, `break`).
                - 행별 유효 콘텐츠 길이 기록:
                    - 패딩이 발생한 행은 `row_lengths.append(content_len)`을 기록하고, 꽉 찬 행은 `row_lengths.append(row_capacity)`를 기록합니다.
                    - 완성된 각 행을 `row_capacity` 길이로 안전 슬라이싱하여 `rows`와 `mask_rows`에 추가합니다.
            - outputs:
                - 패딩이 포함되어 $T + 1$ 길이로 맞춰진 2D 토큰 ID 리스트 `rows` (`List[List[int]]`, 형태: $B \times (T + 1)$)
                - 대응하는 2D 이진 마스크 리스트 `mask_rows` (`List[List[int]]`, 형태: $B \times (T + 1)$)
                - 각 행의 패딩 시작 직전 실제 토큰 경계를 나타내는 1D 정수 리스트 `row_lengths` (`List[int]`, 길이: $B$)
        6. Loss Mask Target Filtering (targets[mask == 0] = -1, scripts/chat_sft.py)
            - inputs: 
                - 이전 단계에서 패킹 완료된 토큰 ID 리스트 `rows` (크기: $B \times (T + 1)$) 및 마스크 리스트 `mask_rows` (크기: $B \times (T + 1)$)
                - 실행 디바이스 (`device`, CUDA/CPU)
            - algorithm:
                - 텐서화 및 자기회귀 시프트 슬라이싱 (Shifted Slicing):
                    1. `rows`를 2D 텐서 `batch_tensor` (타입: `torch.long`, 크기: $B \times (T + 1)$)로 변환합니다.
                    2. 입력 시퀀스 $x$ 는 앞부분 $T$ 개 토큰으로 슬라이싱하여 `inputs = batch_tensor[:, :-1]` (크기: $B \times T$, 타입: `torch.int32`)에 할당합니다.
                    3. 정답 타깃 시퀀스 $y$ 는 뒷부분 $T$ 개 토큰으로 슬라이싱하여 `targets = batch_tensor[:, 1:]` (크기: $B \times T$, 타입: `torch.int64`)에 할당합니다.
                - 마스크 텐서 시프트 정렬:
                    - `mask_rows`를 2D 텐서 `mask_tensor` (타입: `torch.int8`, 크기: $B \times (T + 1)$)로 변환합니다.
                    - 타깃 텐서 $y$ ($x_{1:T}$)의 위치와 일치하도록 마스크도 1칸 시프트 슬라이싱하여 `mask_targets = mask_tensor[:, 1:]` (크기: $B \times T$)를 추출합니다.
                - 지도 손실 제외 토큰 무시 인덱스 할당 (Supervised Mask Target Filtering):
                    - `targets[mask_targets == 0] = -1` 불리언 인덱싱 연산을 수행합니다.
                    - 사용자 프롬프트(`<|user_start|>` ~ `<|user_end|>`), 시스템 및 특수 제어 토큰, 파이썬 인터프리터 외부 출력값(`<|output_start|>` ~ `<|output_end|>`) 등 어시스턴트가 직접 생성하지 않은 모든 위치(`mask == 0`)의 타깃 값을 `-1`로 강제 치환합니다.
                    - `-1`은 PyTorch 교차 엔트로피 손실 함수(`F.cross_entropy`)의 기본 무시 인덱스(`ignore_index = -1`)이므로, 역전파 과정에서 그래디언트가 계산되지 않고 오직 순수 어시스턴트 답변 토큰에 대해서만 학습 손실이 반영됩니다.
            - outputs:
                - 모델 순방향 입력 텐서 `inputs` (`torch.Tensor`, 크기: $B \times T$, 타입: `torch.int32`)
                - 어시스턴트 정답 외 비지도 토큰 위치가 `-1`로 필터링된 타깃 텐서 `targets` (`torch.Tensor`, 크기: $B \times T$, 타입: `torch.int64`)
        7. Padding Token Target Masking (targets[:, content_len-1:] = -1, scripts/chat_sft.py)
            - inputs: 
                - 이전 단계에서 비지도 마스크 토큰이 1차 필터링된 정답 타깃 텐서 `targets` (크기: $B \times T$, 타입: `torch.int64`)
                - 각 행별 패딩 제외 실제 유효 콘텐츠 길이를 담은 리스트 `row_lengths` (길이: $B$)
                - 목표 행 용량 `row_capacity = T + 1`
            - algorithm:
                - 행별 패딩 발생 여부 탐색:
                    - 배치 내의 각 행($i = 0 \dots B-1$)에 대해 실제 유효 콘텐츠 길이 `content_len = row_lengths[i]`를 확인합니다.
                    - `content_len < row_capacity`인 경우, 해당 행의 뒷부분에 공간 부족으로 인한 `<|bos|>` 패딩 토큰이 채워졌음을 의미합니다.
                - 패딩 경계 오프셋 계산 및 타깃 마스킹 (Slice Masking):
                    - 타깃 텐서 `targets`는 원본 `rows`에서 1칸 시프트(`[:, 1:]`)된 상태이므로, 원본 시퀀스의 `content_len` 번째 토큰(즉, 첫 번째 패딩 토큰)을 예측해야 하는 타깃 인덱스는 `content_len - 1` 위치부터 시작합니다.
                    - 유효 텍스트의 마지막 토큰이 불필요하게 뒤따라오는 첫 번째 패딩 토큰(`<|bos|>`)을 예측하도록 학습되는 현상을 차단하기 위해, 슬라이싱 범위를 `content_len - 1`부터 끝까지 잡고 모두 `-1`로 강제 덮어씁니다:
                        - `targets[i, content_len - 1 :] = -1`
                - 역전파 완전 배제:
                    - 이를 통해 행 뒤쪽에 덧붙여진 모든 패딩 토큰과 패딩으로 향하는 전이 확률 예측에 대해 교차 엔트로피 손실(`ignore_index = -1`)이 완벽히 무시되어 모델 파라미터 업데이트에 아무런 영향을 주지 않습니다.
            - outputs: 유효한 어시스턴트 답변 위치만 정수 토큰 ID로 남아있고, 패딩 구간 및 비지도 구간 전체가 `-1`로 완전 마스킹된 최종 정답 타깃 텐서 `targets` (크기: $B \times T$, 타입: `torch.int64`)
        8. GPU Host-to-Device Transfer (scripts/chat_sft.py)
            - inputs: 
                - 이전 단계들에서 슬라이싱 및 마스킹 처리가 완료된 배치 텐서:
                    - 모델 순방향 입력 텐서 `inputs` (크기: $B \times T$, 타입: `torch.int32`)
                    - 비지도 토큰 및 패딩이 `-1`로 완전 마스킹된 정답 타깃 텐서 `targets` (크기: $B \times T$, 타입: `torch.int64`)
                - 실행 타깃 디바이스 `device` (예: `cuda:0` 등)
                - 고정 메모리 및 비동기 전송 설정 플래그 `use_cuda` (`device_type == "cuda"`)
            - algorithm:
                - 고정 메모리(Pinned Memory) 기반 비동기 HtoD 전송:
                    - CUDA 환경인 경우, 원본 `batch_tensor` 생성 단계에서 이미 `pin_memory=True`로 할당된 페이징 불가능(Page-locked) 호스트 메모리를 활용합니다.
                    - `.to(device=device, non_blocking=use_cuda)` 호출을 통해 PCIe 버스 상에서 비동기 DMA(Direct Memory Access) 전송을 수행합니다.
                    - `inputs`는 `torch.int32`로, `targets`는 `torch.int64`로 각각의 필수 데이터 타입에 맞춰 GPU 메모리(VRAM)로 복사됩니다.
                - 연속 메모리 재정렬 (`contiguous`):
                    - 슬라이싱(`[:, :-1]`, `[:, 1:]`)으로 인해 텐서의 스트라이드(Stride)가 분절되었을 가능성을 방지하기 위해 `.contiguous()`를 호출하여 GPU 상에서 완벽히 연속된(Contiguous) 메모리 블록으로 재정렬합니다.
                - 제너레이터 전달 (`yield`):
                    - 최종 준비된 두 개의 2D GPU 텐서 `(inputs, targets)`를 SFT 학습 루프로 산출(`yield`)합니다.
            - outputs: GPU VRAM 상에 준비 완료되어 트랜스포머 모델 및 손실 함수로 직행할 수 있는 배치 텐서 쌍 `(inputs, targets)`
                - `inputs`: 순방향 연산용 입력 토큰 ID 텐서 (`torch.Tensor`, 형태: $B \times T$, 디바이스: `cuda`, 타입: `torch.int32`)
                - `targets`: 역전파 손실 계산용 정답 토큰 텐서 (`torch.Tensor`, 형태: $B \times T$, 디바이스: `cuda`, 타입: `torch.int64`, 유효 토큰 외 전 영역은 `-1`로 채움)
2. Transformer Architecture (nanochat/gpt.py)
    - inputs:
        - 모델 순방향 입력 토큰 ID 텐서 `inputs` (gpt.py 내부 파라미터명: `idx`, 형태: $B \times T$, 타입: `torch.int32` 또는 `torch.int64`)
        - (선택 / 학습 시) 역전파 손실 계산용 정답 타깃 텐서 `targets` (`torch.Tensor`, 형태: $B \times T$, 타입: `torch.int64`, `ignore_index = -1` 적용)
        - (선택 / 추론 시) KV 캐시 객체 `kv_cache` (`nanochat.engine.KVCache`)
        - 손실 축약 방식 `loss_reduction` (기본값: `'mean'`)
    - algorithm:
        1. 임베딩 조회: Word Token Embedding (wte, Untied Weights)
            - inputs: 
                - 모델 순방향 입력 토큰 ID 텐서 `idx` (`torch.Tensor`, 형태: $B \times T$ , 타입: `torch.int32` 또는 `torch.int64`, 값 범위: $0 \le \text{idx} < V$ )
                - 단어 토큰 임베딩 룩업 테이블 `self.transformer.wte` (`nn.Embedding(padded_vocab_size, config.n_embd)`)
                    - 패딩된 어휘 크기 $V_{\text{pad}}$ (`padded_vocab_size`): 텐서 코어 및 분산 통신 정렬을 위해 64의 배수로 올림 정렬된 크기 (기본 어휘 크기: $32768$ )
                    - 임베딩 차원 $d_{\text{model}}$ (`n_embd`): 모델의 은닉 상태 벡터 차원 크기 (예: $768$ )
            - algorithm:
                1. 가중치 초기화 및 정밀도 사전 변환 (Weight Initialization & Precision Casting):
                    - 모델 가중치 초기화 단계(`init_weights`)에서 평균 0.0, 표준편차 0.8인 정규분포로 임베딩 가중치 행렬을 초기화합니다 (`torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)`).
                    - 옵티마이저가 축소 정밀도(Reduced-precision) 임베딩을 지원하므로, VRAM 대역폭 절약 및 연산 가속을 위해 임베딩 텐서를 연산 정밀도(`COMPUTE_DTYPE`, 주로 `torch.bfloat16`)로 사전 변환합니다 (`self.transformer.wte.to(dtype=COMPUTE_DTYPE)`). 단, GradScaler 호환성 제약이 있는 `torch.float16` 환경에서는 `torch.float32` 상태를 유지합니다.
                2. 가중치 비공유 설계 (Untied Weights Architecture):
                    - 고전적인 GPT-2 스타일의 가중치 공유(Weight Tying)와 달리, 입력 임베딩 테이블(`wte`)과 최종 출력 언임베딩 선형 변환기(`lm_head`)의 가중치를 물리적으로 분리합니다.
                    - 이를 통해 토큰 입력 공간의 의미적 표현 학습과 어휘 예측 로짓 분류 공간의 학습 자유도를 독립적으로 확보하여 표현력을 극대화합니다.
                3. 행 인덱싱을 통한 밀집 표현 인출 (Table Lookup):
                    - 2차원 시퀀스 입력 텐서 `idx`의 각 정수 위치를 인덱스로 활용하여 $V_{\text{pad}} \times d_{\text{model}}$ 크기의 임베딩 행렬로부터 $d_{\text{model}}$ 차원의 밀집 은닉 벡터를 인출합니다.
                    - 수식: $x_{\text{raw}} = \text{wte}[\text{idx}] \in \mathbb{R}^{B \times T \times d_{\text{model}}}$
                4. 활성화 텐서 정밀도 강제 정합 (Activation Precision Guard):
                    - 인출된 은닉 벡터를 연산 정밀도와 엄격히 일치시키기 위해 `x = x.to(COMPUTE_DTYPE)` 변환을 명시적으로 거쳐 다음 정규화 및 트랜스포머 블록 연산 파이프라인으로 전달합니다.
            - outputs: 토큰 정수 ID가 밀집 은닉 표현으로 변환된 순방향 입력 텐서 $x_{\text{raw}}$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ , 타입: `COMPUTE_DTYPE` [기본 `torch.bfloat16`])
        2. 초기 정규화: Root Mean Square Normalization (RMSNorm)
            - inputs: 
                - 이전 단계에서 인출된 원시 임베딩 텐서 $x_{\text{raw}}$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ , 타입: `COMPUTE_DTYPE` [기본 `torch.bfloat16`])
                - 정규화 대상 차원 크기: 마지막 은닉 차원 $d = d_{\text{model}}$ (`x.size(-1)`)
            - algorithm:
                1. 무모수 RMSNorm 함수 정의 (`norm(x)`):
                    - 전통적인 LayerNorm과 달리 평균 이동(Centering / Mean Subtraction) 연산과 학습 가능한 아핀 파라미터(Learnable Scale/Gain $\gamma$ 및 Shift/Bias $\beta$)를 완전히 배제한 간소화된 RMSNorm을 적용합니다.
                    - 구현: `F.rms_norm(x, (x.size(-1),))`
                2. 제곱평균제곱근(Root Mean Square) 계산:
                    - 각 토큰별 은닉 상태 벡터 $x \in \mathbb{R}^{d_{\text{model}}}$ 에 대해 원소들의 제곱 평균의 제곱근을 구합니다: $$\text{RMS}(x) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{j=1}^{d_{\text{model}}} x_j^2 + \epsilon}$$ (여기서 $\epsilon$ 은 수치 안정성을 위한 미소값, PyTorch `F.rms_norm` 기본값 $1\text{e-}5$ 적용)
                3. 단위 스케일 재스케일링 (Unit-RMS Scaling):
                    - 원시 임베딩 벡터의 각 원소를 해당 벡터의 $\text{RMS}(x)$ 로 나누어 은닉 벡터의 평균적인 크기(Scale)를 $1.0$ 수준으로 균일하게 맞춥니다: $$\bar{x} = \frac{x}{\text{RMS}(x)}$$
                4. 초기화 스케일 안정화 및 연산 효율성 확보:
                    - 앞 단계에서 큰 분산($\text{std}=0.8$)으로 무작위 초기화된 원시 임베딩 벡터들을 균일한 스케일로 압축하여, 이후 이어지는 Smear 결합 및 잔차 스트림 전파 과정에서 발생할 수 있는 신호 폭발을 방지합니다.
                    - 학습 파라미터가 전혀 없는 순수 수학적 연산(Parameter-free)으로 동작하며, `bfloat16` 정밀도로 직접 실행되어 추가 메모리 할당 없이 고속으로 처리됩니다.
            - outputs: RMS 스케일이 $1.0$ 단위로 표준화된 초기 은닉 상태 텐서 $x$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ , 타입: `COMPUTE_DTYPE` [기본 `torch.bfloat16`])
        3. 이전 토큰 임베딩 결합: Smear Mechanism (Gated Bigram Embedding Mixing)
            - inputs: 
                - 초기 RMSNorm을 통과한 정규화 임베딩 텐서 $x$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ , 타입: `COMPUTE_DTYPE` [기본 `torch.bfloat16`])
                - Smear 투영 게이트 레이어 `self.smear_gate` (`Linear(24, 1, bias=False)`)
                - 학습 가능한 Smear 전역 스케일러 `self.smear_lambda` (`nn.Parameter(torch.zeros(1))`)
                - (선택 / 추론 시) KV 캐시 객체 `kv_cache` 내부에 보관된 직전 스텝 임베딩 상태 `kv_cache.prev_embedding` (형태: $B \times 1 \times d_{\text{model}}$ )
            - algorithm:
                1. 메커니즘 개념 및 목적:
                    - 무거운 셀프 어텐션 블록을 통과하기 전, 인접한 직전 토큰( $t-1$ )의 초기 임베딩 정보를 현재 토큰( $t$ )의 임베딩에 동적으로 주입하여 저비용으로 바이그램(Bigram-like) 문맥 정보를 조기 형성합니다.
                    - 학습 초기 충격을 방지하기 위해 `self.smear_lambda`는 $0.0$ 으로 초기화되어(Identity mapping으로 시작) 학습 진행에 따라 필요한 결합량이 점진적으로 활성화됩니다.
                2. 채널 슬라이싱 및 동적 게이트 연산:
                    - 전체 은닉 차원($d_{\text{model}}$) 중 앞쪽 24개 채널($x_{:, :, :24}$)만을 추출하여 입력으로 사용함으로써 파라미터 수와 연산량을 최소화합니다.
                    - 입력 의존적 시그모이드 게이트와 학습 가능한 스칼라 $\lambda_{\text{smear}}$ 를 곱해 토큰별 결합 비율 $g_t$ 를 계산합니다: $$g_t = \lambda_{\text{smear}} \cdot \sigma(\text{smear\_gate}(x_{t, :24})) \quad (\text{형태: } B \times 1 \times 1)$$
                3. 실행 모드별 분기 처리:
                    - 분기 A: 학습 및 순수 생성 모드 (`kv_cache is None`)
                        1. 전체 시퀀스 길이 검증: $T > 1$ 조건을 단언 검증합니다.
                        2. 위치별 게이트 계산: $1$ 번 인덱스부터 끝까지의 토큰에 대해 게이트를 계산합니다 (`gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))`).
                        3. 시프트 결합 및 연결: 시작 토큰($t=0$)은 결합 대상 이전 토큰이 없으므로 원본을 유지하고, 나머지 위치($t \ge 1$)는 직전 토큰($x_{:, :-1, :}$)에 게이트를 곱해 가산한 뒤 시퀀스 축으로 결합합니다: $$x_{0} = x_{0}$$ $$x_{t} = x_{t} + g_t \odot x_{t-1} \quad (t = 1 \dots T-1)$$
                            - 구현: `torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)`
                    - 분기 B: KV 캐시 기반 고속 추론 모드 (`kv_cache is not None`)
                        1. 직전 토큰 캐시 인출 및 현재 토큰 캐싱: `kv_cache.prev_embedding`에서 직전 임베딩을 읽어오고, 현재 시퀀스의 마지막 토큰 임베딩($x_{:, -1:, :}$)을 다음 생성을 위한 캐시로 갱신 보관합니다.
                        2. 프롬프트 프리필 단계 ($T > 1$): 학습 모드와 동일하게 $1$ 번 인덱스부터 $T-1$ 번 인덱스까지 슬라이싱을 통해 일괄 바이그램 결합을 수행합니다.
                        3. 단일 토큰 디코딩 단계 ($T = 1$): 캐시된 직전 임베딩($x_{\text{pre\_smear}}$)이 존재하는 경우, 단일 토큰에 대해 게이트를 계산하고 즉시 가산합니다: $$x = x + g \odot x_{\text{pre\_smear}}$$
            - outputs: 직전 토큰의 의미 정보가 동적으로 블렌딩된 바이그램 보강 은닉 상태 텐서 $x$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ , 타입: `COMPUTE_DTYPE` [기본 `torch.bfloat16`])

        4. 초기 상태 보존: $x_0$ State Caching
            - inputs: 
                - Smear 결합을 거친 초기 정규화 은닉 텐서 $x$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ , 타입: `COMPUTE_DTYPE` [기본 `torch.bfloat16`])
            - algorithm:
                1. 초기 은닉 표현 스냅샷 보존:
                    - 각 트랜스포머 레이어를 거치며 잔차 스트림이 누적 왜곡되기 전, 토큰의 원시 의미 정보를 온전히 보존하기 위해 현재 상태를 $x_0$ 참조 텐서로 캐싱합니다 (`x0 = x`).
                    - $x_0$ 은 이후 모든 트랜스포머 블록의 초입에서 감쇠형 계수($\lambda_{x_0}^{(i)}$)와 결합되어 깊은 레이어에서도 초기 의미 표현이 망각되지 않도록 직접 주입됩니다.
            - outputs: 불변 참조 상태로 캐싱된 초기 임베딩 텐서 $x_0$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ , 타입: `COMPUTE_DTYPE`)

        5. Transformer Block Stack Loop (Layer $i = 0 \dots N-1$)
            1. 잔차 스케일링 및 초기 상태 합성: Layer-wise Residual & $x_0$ Blending (resid_lambdas, x0_lambdas)
                - inputs: 
                    - 현재 레이어 입력 은닉 텐서 $x$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ )
                    - 초기 캐싱 텐서 $x_0$ (`torch.Tensor`, 형태: $B \times T \times d_{\text{model}}$ )
                    - 학습 가능한 레이어별 잔차 스케일러 `self.resid_lambdas[i]` (초기값: $1.15 - 0.10 \times \frac{i}{N-1}$ )
                    - 학습 가능한 레이어별 $x_0$ 주입 스케일러 `self.x0_lambdas[i]` (초기값: $0.20 - 0.15 \times \frac{i}{N-1}$ )
                - algorithm:
                    1. 동적 잔차 스케일링 및 $x_0$ 합성:
                       - 레이어 인덱스 $i$ 에 따라 잔차 상태 $x$ 와 초기 상태 $x_0$ 을 선형 결합합니다:
                         $$x = \lambda_{\text{resid}}^{(i)} \cdot x + \lambda_{x_0}^{(i)} \cdot x_0$$
                       - 초기 레이어는 강한 잔차 보존($1.15$)과 높은 $x_0$ 결합($0.20$)으로 시작하여, 깊은 레이어로 갈수록 완만하게 감쇠(각각 $1.05$, $0.05$)하도록 초기화되어 레이어별 표현 학습 속도를 차등화합니다.
                - outputs: 스케일링 및 초기 상태가 합성된 은닉 텐서 $x$ (형태: $B \times T \times d_{\text{model}}$ )

            2. Value Embedding 조회: ResFormer Value Embedding (value_embeds, 교차 레이어)
                - inputs: 
                    - 원본 입력 정수 토큰 ID 텐서 `idx` (`torch.Tensor`, 형태: $B \times T$ )
                    - Value Embedding 테이블 딕셔너리 `self.value_embeds` (`nn.ModuleDict`)
                    - 현재 레이어 인덱스 $i$ , 전체 레이어 수 $N$ (`config.n_layer`)
                - algorithm:
                    1. 교차 레이어 배치 판별 (`has_ve`):
                       - `layer_idx % 2 == (n_layer - 1) % 2` 조건을 검사하여, 끝 레이어를 포함해 1칸씩 건너뛰는 교차 레이어에만 Value Embedding 모듈을 할당합니다.
                    2. 전용 토큰 가치 임베딩 룩업:
                       - 해당 레이어에 `str(i)` 키가 존재하는 경우, $V_{\text{pad}} \times d_{\text{kv}}$ 크기의 테이블에서 토큰별 고유 Value 벡터를 인출합니다 (`ve = self.value_embeds[str(i)](idx).to(x.dtype)`).
                       - $d_{\text{kv}} = n_{\text{kv\_head}} \times d_{\text{head}}$ 로 설정되어 Attention Value 프로젝션과 동일한 차원을 가집니다.
                       - 해당하지 않는 비할당 레이어에서는 `ve = None`으로 바이패스합니다.
                - outputs: 인출된 Value Embedding 텐서 `ve` (`torch.Tensor` 또는 `None`, 형태: $B \times T \times d_{\text{kv}}$ )

            3. Attention Sub-block
                1. Pre-Attention Normalization (RMSNorm)
                    - inputs: 이전 단계의 은닉 텐서 $x$ (형태: $B \times T \times d_{\text{model}}$ )
                    - algorithm: 무모수 RMSNorm을 적용하여 입력 벡터의 스케일을 $1.0$ 으로 정규화합니다 (`norm(x) = x / RMS(x)`).
                    - outputs: 정규화된 은닉 텐서 $x_{\text{norm}}$ (형태: $B \times T \times d_{\text{model}}$ )
                2. Q/K/V Linear Projection (c_q, c_k, c_v, Grouped-Query Attention: GQA)
                    - inputs: 정규화된 은닉 텐서 $x_{\text{norm}}$
                    - algorithm:
                        1. 바이어스 없는 선형 투영 레이어를 통해 Q, K, V를 단일 연산으로 인출합니다:
                           - Query: `self.c_q(x_norm).view(B, T, n_head, head_dim)`
                           - Key: `self.c_k(x_norm).view(B, T, n_kv_head, head_dim)`
                           - Value: `self.c_v(x_norm).view(B, T, n_kv_head, head_dim)`
                        2. Flash Attention 3(FA3)의 네이티브 메모리 레이아웃인 $(B, T, H, D)$ 포맷을 그대로 유지하여 전치(Transpose) 오버헤드를 완전히 제거합니다.
                    - outputs: $q$ ($B \times T \times H_q \times D$ ), $k$ ($B \times T \times H_{kv} \times D$ ), $v$ ($B \times T \times H_{kv} \times D$ )
                3. ResFormer Value Gating & Injection ($V + 3 \cdot \sigma(\text{gate}) \odot \text{VE}$)
                    - inputs: Value 텐서 $v$ , Value Embedding 텐서 `ve` , 정규화 전 은닉 텐서 $x$
                    - algorithm:
                        1. `ve is not None` 조건일 경우 은닉 텐서 $x$ 의 앞쪽 12개 채널($x_{:, :, :12}$)을 추출합니다.
                        2. 게이트 레이어 `self.ve_gate`를 통해 헤드별 게이트 값을 산출하고, 범위를 $(0, 3)$ 으로 스케일링합니다:
                           $$\text{gate} = 3 \cdot \sigma(\text{ve\_gate}(x_{:, :, :12})) \quad (\text{형태: } B \times T \times H_{kv})$$
                        3. 투영된 원본 Value 벡터에 입력 의존적 게이트가 곱해진 Value Embedding을 가산 주입합니다:
                           $$v = v + \text{gate.unsqueeze}(-1) \odot \text{ve}$$
                    - outputs: 토큰 고유 Value 정보가 합성된 최종 Value 텐서 $v$ ($B \times T \times H_{kv} \times D$ )
                4. 회전 위치 임베딩 적용: Rotary Position Embedding (RoPE)
                    - inputs: 쿼리 $q$ , 키 $k$ , 사전 계산된 코사인/사인 버퍼 `cos_sin`
                    - algorithm:
                        1. KV 캐시 위치 오프셋 $T_0$ 에 맞춰 RoPE 텐서를 슬라이싱합니다.
                        2. $q$ 와 $k$ 의 마지막 차원을 2분할($x_1, x_2$)하여 회전 변환을 수행합니다:
                           $$y_1 = x_1 \cdot \cos + x_2 \cdot \sin, \quad y_2 = x_1 \cdot (-\sin) + x_2 \cdot \cos$$
                           $$\text{RoPE}(x) = [y_1, y_2]$$
                    - outputs: 상대적 위치 정보가 인코딩된 $q_{\text{rope}}, k_{\text{rope}}$
                5. 쿼리/키 정규화: QK Normalization (RMSNorm)
                    - inputs: 위치 인코딩된 $q_{\text{rope}}, k_{\text{rope}}$
                    - algorithm: 학습 중 어텐션 로짓 폭발 및 수치 불안정성을 방지하기 위해 $q$ 와 $k$ 의 마지막 채널 차원에 무모수 RMSNorm을 적용합니다 (`q = norm(q)`, `k = norm(k)`).
                    - outputs: 정규화된 쿼리/키 텐서 $q_{\text{norm}}, k_{\text{norm}}$
                6. 어텐션 엔트로피 스케일링: Attention Sharpness Scaling ($1.2 \times Q, 1.2 \times K$)
                    - inputs: 정규화된 쿼리/키 텐서 $q_{\text{norm}}, k_{\text{norm}}$
                    - algorithm: 어텐션 확률 분포의 엔트로피를 낮추고 샤프한 어텐션 패턴을 형성하기 위해 $Q$ 와 $K$ 에 각각 $1.2$ 스케일러를 곱합니다 ($q = 1.2 \cdot q, k = 1.2 \cdot k$, 합성 스케일링 효과: $1.44 \times$).
                    - outputs: 샤프니스 스케일링이 적용된 쿼리/키 텐서 $q, k$
                7. 인과적 슬라이딩 윈도우 어텐션 연산: Causal Sliding Window Attention (SWA, FA3 / SDPA Fallback)
                    - inputs: $q, k, v$ , 현재 레이어 윈도우 크기 `window_size` (예: "SSSL" 패턴에 따라 $S=(768, 0)$ 또는 $L=(2048, 0)$), `kv_cache`
                    - algorithm:
                        1. 학습 모드 (`kv_cache is None`):
                           - `flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)`를 호출하여 인과적 마스크 및 슬라이딩 윈도우가 결합된 어텐션을 고속 수행합니다.
                        2. 추론 모드 (`kv_cache is not None`):
                           - `flash_attn.flash_attn_with_kvcache`를 호출하여 현재 $k, v$ 를 캐시에 인플레이스(In-place) 갱신하고 슬라이딩 윈도우를 적용해 어텐션을 산출합니다.
                           - 마지막 레이어 도달 시 캐시 위치를 자동 전진시킵니다 (`kv_cache.advance(T)`).
                    - outputs: 어텐션 집약 텐서 $y_{\text{attn}}$ ($B \times T \times H_q \times D$ )
                8. Attention Output Projection (c_proj)
                    - inputs: 어텐션 결과 텐서 $y_{\text{attn}}$
                    - algorithm:
                        1. 헤드 차원을 은닉 차원으로 결합 평탄화합니다 (`contiguous().view(B, T, -1)`).
                        2. 0으로 초기화된 프로젝션 선형 변환(`c_proj`)을 거쳐 잔차 스트림 차원으로 매핑합니다.
                    - outputs: 투영된 어텐션 출력 텐서 $y$ ($B \times T \times d_{\text{model}}$ )
                9. 잔차 연결: Residual Addition
                    - inputs: 서브블록 입력 텐서 $x$ , 투영 출력 텐서 $y$
                    - algorithm: $x = x + y$ 연산을 통해 어텐션 블록의 연산 결과를 잔차 스트림에 누적합니다.
                    - outputs: 어텐션 결과가 결합된 은닉 텐서 $x$ ($B \times T \times d_{\text{model}}$ )

            4. MLP Sub-block
                1. Pre-MLP Normalization (RMSNorm)
                    - inputs: 어텐션 잔차 텐서 $x$
                    - algorithm: 무모수 RMSNorm을 적용하여 은닉 벡터를 단위 스케일로 정규화합니다 (`norm(x)`).
                    - outputs: 정규화된 은닉 텐서 $x_{\text{norm}}$ ($B \times T \times d_{\text{model}}$ )
                2. Linear Expansion (c_fc: $d_{\text{model}} \to 4d_{\text{model}}$)
                    - inputs: 정규화된 은닉 텐서 $x_{\text{norm}}$
                    - algorithm: 선형 투영 레이어를 통해 은닉 차원을 4배로 확장합니다 (`x = self.c_fc(x)`).
                    - outputs: 확장된 은닉 텐서 ($B \times T \times 4d_{\text{model}}$ )
                3. 제곱 정류 선형 유닛 활성화: Squared ReLU Activation ($\text{ReLU}^2$)
                    - inputs: 4배 확장된 은닉 텐서
                    - algorithm:
                        1. 음수 성분을 제거하는 ReLU 활성화를 적용합니다 (`F.relu(x)`).
                        2. 결과를 제곱하여 활성화 크기에 비선형 가중치를 부여합니다 (`x.square()`):
                           $$\text{ReLU}^2(x) = (\max(0, x))^2$$
                    - outputs: 활성화 텐서 ($B \times T \times 4d_{\text{model}}$ )
                4. Linear Down-projection (c_proj: $4d_{\text{model}} \to d_{\text{model}}$)
                    - inputs: $\text{ReLU}^2$ 활성화 텐서
                    - algorithm: 0으로 초기화된 선형 투영 레이어를 통해 원래 모델 차원으로 축소합니다 (`self.c_proj(x)`).
                    - outputs: 축소 투영 텐서 $y_{\text{mlp}}$ ($B \times T \times d_{\text{model}}$ )
                5. 잔차 연결: Residual Addition
                    - inputs: 서브블록 입력 텐서 $x$ , MLP 축소 텐서 $y_{\text{mlp}}$
                    - algorithm: $x = x + y_{\text{mlp}}$ 를 수행하여 MLP 출력을 잔차 스트림에 누적합니다.
                    - outputs: MLP 결과가 결합된 레이어 최종 은닉 텐서 $x$ ($B \times T \times d_{\text{model}}$ )

            5. 중간 레이어 잔차 상태 캐싱: Mid-layer State Caching ($i = N // 2$)
                - inputs: 현재 레이어 완료 후의 은닉 텐서 $x$ , 레이어 인덱스 $i$
                - algorithm: $i == N // 2$ (중간 깊이 레이어)에 도달했을 때, 현재의 잔차 상태를 $x_{\text{backout}}$ 변수에 보관합니다 (`x_backout = x`).
                - outputs: 캐싱된 중간 은닉 상태 $x_{\text{backout}}$ (`torch.Tensor` 또는 `None`, 형태: $B \times T \times d_{\text{model}}$ )

        6. 중간 레이어 잔차 차감: Backout Mechanism ($x - \lambda_{\text{backout}} \cdot x_{\text{mid}}$)
            - inputs: 
                - $N$ 개 레이어를 모두 거친 최종 은닉 텐서 $x$
                - 캐시된 중간 레이어 은닉 상태 $x_{\text{backout}}$
                - 학습 가능한 차감 스칼라 `self.backout_lambda` (초기값: $0.2$ )
            - algorithm:
                1. 저차원 표면적 피처 제거:
                   - 잔차 스트림에서 중간 레이어의 저수준 특징이 최종 예측에 과도하게 간섭하는 것을 배제하기 위해, 학습된 비율만큼 중간 상태를 직접 감산합니다:
                     $$x = x - \lambda_{\text{backout}} \cdot x_{\text{backout}}$$
            - outputs: 저수준 피처가 제거된 정제된 최종 은닉 텐서 $x$ (형태: $B \times T \times d_{\text{model}}$ )

        7. 최종 정규화: Final RMSNorm
            - inputs: Backout 감산이 완료된 은닉 텐서 $x$
            - algorithm: 출력 프로젝션 직전, 무모수 RMSNorm을 적용하여 은닉 벡터의 크기를 $1.0$ 으로 최종 안정화합니다 (`norm(x)`).
            - outputs: 최종 정규화 텐서 $x_{\text{norm}}$ ($B \times T \times d_{\text{model}}$ )

        8. 언임베딩 선형 투영: Unembedding / LM Head Projection (lm_head, Untied Weights)
            - inputs: 최종 정규화 텐서 $x_{\text{norm}}$
            - algorithm:
                1. `wte`와 완전히 독립된 가중치 행렬 `self.lm_head` ($d_{\text{model}} \to V_{\text{pad}}$)를 통해 토큰별 어휘 로짓을 선형 계산합니다 (`logits = self.lm_head(x_norm)`).
            - outputs: 패딩 어휘 차원을 포함하는 미가공 로짓 텐서 `logits` (형태: $B \times T \times V_{\text{pad}}$ )

        9. 패딩 어휘 슬라이싱: Vocab Padding Slicing
            - inputs: 미가공 로짓 텐서 `logits`
            - algorithm: 하드웨어 가속 정렬용으로 패딩되었던 유효하지 않은 어휘 차원($V_{\text{pad}} - V$)을 슬라이싱으로 잘라냅니다 (`logits = logits[..., :self.config.vocab_size]`).
            - outputs: 실제 유효 어휘 사전 크기로 정렬된 로짓 텐서 `logits` (형태: $B \times T \times V$ )

        10. 로짓 소프트 캡핑: Logit Soft-capping ($15 \cdot \tanh(\text{logits} / 15)$)
            - inputs: 슬라이싱된 로짓 텐서 `logits`
            - algorithm:
                1. 수치 안정성을 위해 로짓 텐서를 `torch.float32`로 업캐스팅합니다 (`logits.float()`).
                2. 하이퍼볼릭 탄젠트를 활용하여 로짓의 최대/최소 범위를 $[-15, +15]$ 로 완만하게 캡핑(Clamping)합니다:
                   $$\text{logits}_{\text{capped}} = 15 \cdot \tanh\left(\frac{\text{logits}}{15}\right)$$
                3. 극단적인 로짓 폭발을 방지하여 교차 엔트로피 손실의 이상 과열을 원천 차단합니다.
            - outputs: 범위가 $[-15, +15]$ 로 제한된 FP32 로짓 텐서 `logits` (형태: $B \times T \times V$ )
    - outputs:
        - 학습 모드: 스칼라 손실값 `loss` (`torch.Tensor`, 형태: `()`)
            - 캡핑된 로짓 텐서 `logits`와 정답 타깃 텐서 `targets`를 2D 평탄화한 후 `F.cross_entropy(..., ignore_index=-1, reduction=loss_reduction)`을 적용하여, `-1`로 마스킹된 비지도 및 패딩 구간을 완전히 배제하고 산출된 최종 손실 스칼라입니다.
        - 추론 모드: 다음 토큰 생성용 로짓 텐서 `logits` (`torch.Tensor`, 형태: $B \times T \times V$ 또는 $B \times 1 \times V$)
            - `targets is None`일 때 추가 손실 계산 없이 추론 엔진의 샘플링 파이프라인으로 직행하는 소프트 캡핑 완료 로짓 텐서입니다.

3. Optimization & Training Dynamics (nanochat/optim.py, scripts/base_train.py)
    - inputs:
        - 모델 파라미터 그래디언트 (1D/스칼라/임베딩 및 2D 트랜스포머 가중치 행렬)
        - 학습률 스케줄러 및 옵티마이저 하이퍼파라미터 (`embedding_lr`, `unembedding_lr`, `matrix_lr`, `scalar_lr`, `weight_decay` 등)
        - 분산 학습 프로세스 그룹 메타데이터 (`ddp_rank`, `ddp_world_size`)
    - algorithm:
        1. 하이퍼파라미터 스케줄링 및 Scaling Laws
            1. Power Law 기반 최적 배치 크기 계산 ($B_{\text{opt}} \propto D^{0.383}$)
                - inputs: 목표 학습 토큰량 $D$ (`target_tokens`), 기준 d12 모델 토큰량 $D_{\text{ref}}$ , 기준 배치 크기 $B_{\text{ref}} = 2^{19} = 524,288$
                - algorithm:
                    1. 거듭제곱 법칙에 따라 최적 배치 크기를 도출합니다:
                       $$B_{\text{predicted}} = B_{\text{ref}} \cdot \left(\frac{D}{D_{\text{ref}}}\right)^{0.383}$$
                    2. 연산 효율성을 위해 가장 가까운 2의 거듭제곱으로 반올림 클램핑합니다:
                       $$B_{\text{opt}} = 2^{\text{round}(\log_2(B_{\text{predicted}}))}$$
                - outputs: 전역 최적 배치 크기 `total_batch_size` (정수, 토큰 단위)

            2. $T_{\text{epoch}}$ 불변 프레임워크 학습률 스케일링 ($\sqrt{B / B_{\text{ref}}}$)
                - inputs: 최적 배치 크기 $B$ , 기준 배치 크기 $B_{\text{ref}}$
                - algorithm: 배치 크기 확장에 따른 표준 제곱근 스케일링 비율을 산출합니다:
                  $$\text{batch\_lr\_scale} = \sqrt{\frac{B}{B_{\text{ref}}}}$$
                  모든 기본 학습률(Matrix, Embedding, Unembedding)에 이 계수를 일괄 곱합니다.
                - outputs: 배치 스케일링 보정 계수 `batch_lr_scale` (실수)

            3. $T_{\text{epoch}}$ 불변 프레임워크 가중치 감쇠 스케일링 ($\lambda_{\text{ref}} \cdot \sqrt{B/B_{\text{ref}}} \cdot (D_{\text{ref}}/D)$)
                - inputs: 기준 가중치 감쇠 $\lambda_{\text{ref}} = 0.28$ , 배치 비율 $B / B_{\text{ref}}$ , 토큰 비율 $D_{\text{ref}} / D$
                - algorithm: 유효 에포크 주기 $T_{\text{epoch}} = \frac{B}{\eta \cdot \lambda \cdot D}$ 가 모델 크기 및 토큰량 변화에도 일정하게 유지되도록 가중치 감쇠를 보정합니다:
                  $$\lambda_{\text{scaled}} = \lambda_{\text{ref}} \cdot \sqrt{\frac{B}{B_{\text{ref}}}} \cdot \left(\frac{D_{\text{ref}}}{D}\right)$$
                - outputs: 보정된 Muon 가중치 감쇠율 `weight_decay_scaled`

            4. 모델 폭 비례 AdamW LR 스케일링 ($\propto 1 / \sqrt{d_{\text{model}}}$)
                - inputs: 모델 차원 $d_{\text{model}}$ (`n_embd`), 기준 튜닝 폭 $768$
                - algorithm: 모델 폭 확장에 따른 안정적 전이를 위해 AdamW 파라미터(임베딩, 언임베딩)의 학습률을 폭의 제곱근에 반비례하도록 보정합니다:
                  $$\text{dmodel\_lr\_scale} = \sqrt{\frac{768}{d_{\text{model}}}}$$
                - outputs: 폭 보정 스케일러 `dmodel_lr_scale`

            5. Linear Warmup - Constant - Linear Warmdown LR 스케줄링
                - inputs: 현재 스텝 $t$ , 전체 스텝 $T_{\text{total}}$ , 웜업 스텝 $T_{\text{warmup}}=40$ , 웜다운 비율 $R_{\text{warmdown}}=0.65$ , 최종 LR 비율 $F_{\text{final}}=0.05$
                - algorithm:
                    1. Warmup ($t < T_{\text{warmup}}$): $\frac{t+1}{T_{\text{warmup}}}$ 로 선형 증가
                    2. Constant ($T_{\text{warmup}} \le t \le T_{\text{total}} - T_{\text{warmdown}}$): $1.0$ 유지
                    3. Warmdown: $1.0$ 에서 $F_{\text{final}}$ 까지 선형 감소
                - outputs: 현재 스텝의 학습률 승수 `lrm` ($0.05 \sim 1.0$)

            6. Cosine Decay Muon Weight Decay 스케줄링
                - inputs: 현재 스텝 $t$ , 전체 스텝 $T_{\text{total}}$ , 스케일링된 가중치 감쇠 $\lambda_{\text{scaled}}$
                - algorithm: 코사인 함수를 통해 학습 종료 시점까지 가중치 감쇠를 $0$ 으로 점진적 소멸시킵니다:
                  $$\lambda(t) = \lambda_{\text{scaled}} \cdot \frac{1}{2} \left(1 + \cos\left(\frac{\pi \cdot t}{T_{\text{total}}}\right)\right)$$
                - outputs: 현재 스텝의 Muon 가중치 감쇠율

            7. Warmup-Warmdown Muon Momentum 스케줄링
                - inputs: 현재 스텝 $t$ , 웜다운 진입 스텝 $t_{\text{wd\_start}}$
                - algorithm:
                    1. 초반 400스텝: $0.85$ 에서 $0.97$ 로 선형 웜업
                    2. 중반: $0.97$ 고정 유지
                    3. 웜다운 구간: $0.97$ 에서 $0.90$ 으로 선형 감쇠
                - outputs: 동적 모멘텀 계수 $\mu(t)$

        2. 파라미터 분기 및 Mixed Optimizer (MuonAdamW)
            1. 분기 1: 1D/스칼라/임베딩 (AdamW)
                1. ZeRO-2 스타일 그래디언트 Reduce-Scatter (대형 텐서) / All-Reduce (소형 텐서)
                    - inputs: 1D/스칼라/임베딩 그래디언트 텐서 `grad` , 프로세스 그룹 정보 ($W_{\text{size}}$, $\text{Rank}$)
                    - algorithm:
                        - 원소 수 $< 1024$ (소형): 비동기 `dist.all_reduce(grad, op=AVG)`를 호출하여 전 랭크가 동일 그래디언트를 복제 수신합니다.
                        - 원소 수 $\ge 1024$ (대형): `grad.shape[0]`를 $W_{\text{size}}$ 로 분할하여 비동기 `dist.reduce_scatter_tensor`를 실행하고, 각 랭크는 자신의 샤드 $\frac{1}{W_{\text{size}}}$ 만 수신합니다.
                    - outputs: 통신 퓨처 객체 `future` 및 로컬 그래디언트 슬라이스 `grad_slice`
                2. Fused AdamW Step (Decoupled Weight Decay, 1st/2nd 모멘텀, Bias Correction)
                    - inputs: 가중치 슬라이스 $p$ , 그래디언트 슬라이스 $g$ , 1차 모멘텀 $m$ , 2차 모멘텀 $v$ , 스텝 수 $t$
                    - algorithm:
                        1. FP32로 업캐스팅하여 가중치 감쇠를 선적용합니다: $p \leftarrow p \cdot (1 - \eta \cdot \lambda)$
                        2. 이동평균을 갱신합니다: $m \leftarrow \beta_1 m + (1-\beta_1) g$ , $v \leftarrow \beta_2 v + (1-\beta_2) g^2$
                        3. 편향 보정을 적용하고 파라미터를 업데이트합니다:
                           $$\hat{m} = \frac{m}{1 - \beta_1^t}, \quad \hat{v} = \frac{v}{1 - \beta_2^t}$$
                           $$p \leftarrow p - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \cdot \hat{m}$$
                        - `@torch.compile(fullgraph=True)` 기반의 단일 융합 커널(`adamw_step_fused`)로 실행됩니다.
                    - outputs: 갱신된 로컬 파라미터 슬라이스 $p$ 및 상태 텐서 ($m, v$)
                3. 파라미터 All-Gather 및 복원
                    - inputs: 갱신된 로컬 파라미터 슬라이스 $p_{\text{slice}}$
                    - algorithm: 대형 텐서의 경우 비동기 `dist.all_gather_into_tensor(p, p_slice)`를 호출하여 분산 갱신된 슬라이스를 모아 전체 파라미터 텐서를 완전 복원합니다.
                    - outputs: 전 랭크에 동기화 복원된 전체 파라미터 텐서 $p$

            2. 분기 2: 2D 가중치 행렬 (Muon)
                1. 동형 텐서 스택 및 분산 청크 Reduce-Scatter
                    - inputs: 동일 형태(Shape)를 갖는 $K$ 개의 2D 가중치 그래디언트 목록
                    - algorithm:
                        1. $K$ 개 그래디언트를 단일 $(K, H, W)$ 텐서로 스택하고 월드 크기 배수로 제로 패딩합니다.
                        2. `dist.reduce_scatter_tensor`를 호출하여 각 랭크에 $\text{chunk\_size} = \lceil K / W_{\text{size}} \rceil$ 개 텐서 슬라이스를 할당합니다.
                    - outputs: 각 랭크가 전담 소유한 그래디언트 텐서 청크 `grad_chunk`
                2. Nesterov Momentum 업데이트
                    - inputs: 그래디언트 청크 $G$ , 1차 모멘텀 버퍼 $M$ , 모멘텀 계수 $\mu$
                    - algorithm:
                        $$M \leftarrow \mu \cdot M + (1 - \mu) \cdot G$$
                        $$X \leftarrow (1 - \mu) \cdot G + \mu \cdot M = G.\text{lerp}(M, \mu)$$
                    - outputs: 네스테로프 가속 그래디언트 $X$
                3. MuonEq 행 노름 균형화 (Row Norm Equilibration)
                    - inputs: 네스테로프 그래디언트 $X$
                    - algorithm:
                        1. 행별 L2 노름을 구하고 평균 행 노름 타깃을 계산합니다:
                           $$\text{target} = \frac{\|X\|_F}{\sqrt{\text{행 개수}}}, \quad \text{row\_norm} = \|X_{i, :}\|_2$$
                        2. 행 벡터를 재스케일링하여 스펙트럼 불균형을 사전에 완화합니다:
                           $$X \leftarrow X \cdot \frac{\text{target}}{\text{row\_norm} + 1\text{e-}6}$$
                    - outputs: 행 노름이 균일화된 텐서 $X$
                4. Polar Express 5차 부호 반복법 (Orthogonalization)
                    - inputs: 균형화된 텐서 $X$
                    - algorithm:
                        1. 스펙트럼 반경 스케일링: $X \leftarrow \frac{X}{\|X\|_F \cdot 1.01 + 1\text{e-}6}$
                        2. 5차 부호 반복 (극점 근사 최적화 계수 $a, b, c$ 적용 5회 루프):
                           - Wide Matrix ($H \le W$):
                             $$A = X X^T, \quad B = b A + c A^2, \quad X \leftarrow a X + B X$$
                           - Tall Matrix ($H > W$):
                             $$A = X^T X, \quad B = b A + c A^2, \quad X \leftarrow a X + X B$$
                    - outputs: 준직교화(Semi-orthogonal)된 업데이트 행렬 $O$
                5. Muon+ 준직교 재정규화 (Snap Frobenius Norm)
                    - inputs: 직교화 텐서 $O$
                    - algorithm: 부호 반복의 미수렴 오차를 보정하기 위해 프로베니우스 노름을 엄밀한 직교 행렬 이론치로 강제 스냅합니다:
                      $$O \leftarrow O \cdot \frac{\sqrt{\min(H, W)}}{\|O\|_F + 1\text{e-}6}$$
                    - outputs: 노름이 보정된 직교 업데이트 텐서 $O$
                6. NorMuon 컬럼/뉴런 분산 축소 (Variance Reduction)
                    - inputs: 직교 업데이트 텐서 $O$ , 2차 분산 버퍼 $v$
                    - algorithm:
                        1. 축소 차원 기준 제곱 평균 $v_{\text{mean}} = \text{mean}(O^2)$ 산출
                        2. 버퍼 갱신: $v \leftarrow \beta_2 v + (1 - \beta_2) v_{\text{mean}}$
                        3. 뉴런별 업데이트 크기를 균일화하는 보정 스케일 적용:
                           $$O \leftarrow O \cdot \frac{v^{-1/2}}{\|O \cdot v^{-1/2}\|_F} \cdot \|O\|_F$$
                    - outputs: 뉴런 간 분산이 정규화된 최종 그래디언트 $G_{\text{final}}$
                7. Cautious Weight Decay 및 파라미터 업데이트
                    - inputs: 가중치 $W$ , 정규화된 그래디언트 $G_{\text{final}}$ , 학습률 $\eta$ , 감쇠율 $\lambda$
                    - algorithm:
                        1. Cautious 마스크 계산: $\text{mask} = (G_{\text{final}} \odot W \ge 0)$ (업데이트 방향과 파라미터 부호가 일치할 때만 가중치 감쇠 허용)
                        2. 파라미터 동시 업데이트:
                           $$W \leftarrow W - \eta \cdot G_{\text{final}} - \eta \cdot \lambda \cdot W \odot \text{mask}$$
                    - outputs: 갱신된 로컬 소유 가중치 청크 $W_{\text{owned}}$
                8. 갱신 파라미터 All-Gather 및 개별 텐서 언패킹
                    - inputs: 로컬 갱신 파라미터 청크 $W_{\text{owned}}$
                    - algorithm:
                        1. `dist.all_gather_into_tensor`를 통해 전체 랭크의 청크를 단일 스택 버퍼로 수집합니다.
                        2. `torch._foreach_copy_`를 호출하여 수집된 스택 텐서를 개별 모듈 파라미터 $K$ 개에 슬라이스 복사(Unpack)합니다.
                    - outputs: 전 랭크에 동기화 완료된 트랜스포머 가중치 2D 행렬들
    - outputs:
        - 1 스텝 역전파 및 최적화가 완료되어 전 분산 랭크에 일괄 동기화된 모델 파라미터 상태 (`model.state_dict()`)
        - ZeRO-2 스타일로 샤딩 분산 저장된 AdamW 및 Muon 옵티마이저 내부 모멘텀 상태 (`optimizer.state_dict()`)

4. Post-Training: SFT & RL Alignment (scripts/chat_sft.py, scripts/chat_rl.py)
    - inputs:
        - 사전학습 완료된 베이스 모델 가중치 체크포인트 (`base_checkpoints/`)
        - 정렬용 대화 및 추론 데이터셋 인스턴스 (`SmolTalk`, `MMLU`, `GSM8K`)
        - 정렬 하이퍼파라미터 (SFT/RL 학습률, 롤아웃 샘플 수 $K=16$ , 샘플링 온도 등)
    - algorithm:
        1. Supervised Fine-Tuning (SFT) Pipeline
            1. Task Mixture 데이터 혼합 (SmolTalk + MMLU + GSM8K) 및 결정론적 셔플링
                - inputs: `SmolTalk` (46만 행), `MMLU` (3회 복제, 30만 행), `GSM8K` (4회 복제, 3.2만 행)
                - algorithm: 가상 길이 기반 오버샘플링 후, 고정 시드(`seed=42`)로 2D 인덱스 맵을 무작위 교차 인터리빙하여 태스크 균형 셔플 데이터셋을 구축합니다.
                - outputs: 약 80만 행 규모의 단일 통합 인터리빙 데이터셋 `train_dataset`
            2. 베이스 모델 모멘텀 버퍼 Warm-start 및 SFT 전용 LR 리셋
                - inputs: 사전학습 체크포인트 옵티마이저 샤드 파일 `optim_*.pt`
                - algorithm:
                    1. 사전학습된 모멘텀 버퍼($m, v$)를 그대로 로드하여 최적화 연속성을 확보합니다.
                    2. 직전 사전학습 종료 시 $0$ 에 수렴했던 학습률을 SFT 베이스 학습률의 $80\%$ (`--init-lr-frac=0.8`)로 강제 덮어써 리셋합니다.
                - outputs: 모멘텀이 보존되고 학습률이 재활성화된 SFT 옵티마이저
            3. BestFit-Pad 기반 시퀀스 패킹 및 Supervised Loss Mask 역전파
                - inputs: 토큰화된 대화 쌍 `(ids, mask)` 버퍼 풀 (크기 100)
                - algorithm:
                    1. Best-Fit 탐색으로 빈 행에 대화를 채우고, 공간 부족 시 자르지 않고 `<|bos|>`로 완전 무손실 패딩합니다.
                    2. 어시스턴트 순수 생성 토큰(`mask == 1`) 외의 사용자 프롬프트 및 패딩 구간 전체를 `targets = -1`로 마스킹하여 오직 모델 응답에 대해서만 그래디언트를 역전파합니다.
                - outputs: 유효 응답에 대해서만 역전파가 수행된 SFT 모델 체크포인트

        2. Reinforcement Learning (RL) Pipeline
            1. GSM8K 프롬프트 생성용 렌더링 (<|assistant_start|> 프라이밍)
                - inputs: GSM8K 학습 문제 딕셔너리 (`question`, `answer`)
                - algorithm: 사용자 질문을 인코딩한 뒤 `<|assistant_start|>` 토큰으로 끝맺어 어시스턴트가 즉시 풀이 생성을 시작할 수 있도록 프라이밍된 입력 토큰 시퀀스를 구성합니다 (`render_for_completion`).
                - outputs: 완결 답변이 제거되고 시작 프롬프트만 남은 토큰 ID 리스트 `tokens`
            2. Engine 기반 Rollout 샘플링 ($K=16$, Top-k / Temperature)
                - inputs: 프라이밍 프롬프트 `tokens` , 샘플링 수 $K=16$ , $T=1.0$ , $\text{top\_k}=50$
                - algorithm: 추론 엔진(`Engine`)을 통해 동일 문제에 대해 서로 다른 난수 시드로 16개의 독립적인 롤아웃 경로를 병렬 생성합니다.
                - outputs: 생성된 16개의 토큰 시퀀스 `generated_token_sequences` 및 마스크 리스트
            3. 환경 보상(Reward) 채점 및 검증
                - inputs: 생성된 텍스트 응답들, 정답 풀이 텍스트
                - algorithm: 정규식으로 `####` 뒤의 최종 수치 정답을 추출하여 일치 여부를 판별하고 이진 보상 $r \in \{0.0, 1.0\}$ 을 부여합니다.
                - outputs: 16개 샘플에 대한 스칼라 보상 텐서 `rewards` ($B=16$ )
            4. 시퀀스 패딩 및 입력/타깃 텐서 정렬
                - inputs: 가변 길이의 16개 롤아웃 시퀀스, 특수 토큰 `<|assistant_end|>`
                - algorithm: 16개 시퀀스의 최대 길이에 맞춰 뒤쪽을 `<|assistant_end|>`로 패딩하고, 입력 $x$ 와 1칸 시프트된 타깃 $y$ 로 정렬한 뒤 프롬프트 및 도구 출력 영역을 `targets = -1`로 마스킹합니다.
                - outputs: 정렬된 텐서 `inputs` ($B \times T$ ), `targets` ($B \times T$ )
            5. DAPO 스타일 토큰 레벨 어드밴티지 계산 ($A_i = r_i - \mu_r$)
                - inputs: 16개 샘플의 보상 텐서 `rewards`
                - algorithm: 표준편차 나누기($z$-score) 없이 그룹 평균 보상 $\mu_r$ 만을 감산하여 어드밴티지를 계산합니다: $A_i = r_i - \mu_r$
                - outputs: 제로 평균화된 어드밴티지 텐서 `advantages` ($B$ )
            6. 간소화된 On-Policy REINFORCE 손실 계산 (KL 페널티 / PPO Clip 배제)
                - inputs: `inputs`, `targets`, `advantages`
                - algorithm:
                    1. 순방향 손실(음의 로그 우도)로부터 각 토큰의 로그 확률 $\log p_\theta(x_t)$ 를 복원합니다 (`logp = -loss2d`).
                    2. 정책 그래디언트 목적식을 도출합니다:
                       $$\mathcal{L}_{\text{PG}} = -\frac{1}{N_{\text{valid}}} \sum_{i, t} \log p_\theta(x_{i, t}) \cdot A_i$$
                       (온폴리시 샘플링이므로 PPO 비율 클리핑 및 참조 모델 KL 페널티를 배제)
                - outputs: 정책 그래디언트 손실 스칼라 `loss`
            7. 유효 토큰 수 정규화 및 정책 그래디언트 업데이트
                - inputs: 손실 `loss` , 유효 타깃 토큰 수 $N_{\text{valid}} = \sum(y \ge 0)$
                - algorithm: 패딩 및 비지도 토큰을 제외한 실제 생성 토큰 수로 엄밀히 정규화한 후 역전파를 수행하고 가중치를 업데이트합니다.
                - outputs: 수학 추론 및 도구 사용 성능이 강화된 정렬 모델 체크포인트
    - outputs:
        - 대화 턴 교차 및 도구 호출 인터페이스를 준수하는 지도 미세조정 가중치 체크포인트 (`chatsft_checkpoints/`)
        - 강화학습 정렬을 통해 자가 연산 및 수학 추론 능력이 극대화된 최종 대화 모델 가중치 체크포인트 (`chatrl_checkpoints/`)

5. Evaluation & Inference Engine (nanochat/loss_eval.py, nanochat/core_eval.py, nanochat/engine.py)
    - inputs:
        - 평가/추론 대상 학습 체크포인트 가중치 및 설정 메타데이터
        - 평가 데이터셋 스트림 (DCLM CORE, GSM8K, 검증 샤드) 또는 대화 CLI 사용자 프롬프트
        - 생성 하이퍼파라미터 (온도 $T$ , `top_k` , `max_tokens` 등)
    - algorithm:
        1. 정량적 평가 메트릭 (Evaluation)
            1. Bits Per Byte 계산: Vocab-Size Invariant NLL 정규화 ($\text{Total Nats} / (\ln 2 \cdot \text{Total Bytes})$)
                - inputs: 모델 예측 손실 텐서 `loss2d` (단위: Nats), 타깃 시퀀스 $y$ , 바이트 길이 매핑 텐서 `token_bytes`
                - algorithm:
                    1. 유효 타깃($y \ge 0$)에 대해 토큰 바이트 길이를 인출하고, 특수 토큰(`<|bos|>`) 및 마스킹 토큰은 바이트 수 $0$ 으로 마스킹합니다.
                    2. 전체 검증 셋에 대해 총 손실 합(Total Nats)과 총 바이트 수 합(Total Bytes)을 독립 누적 집계합니다.
                    3. 밑이 2인 로그로 정규화하여 어휘 사전 크기에 불변인 BPB를 산출합니다:
                       $$\text{BPB} = \frac{\text{Total Nats}}{\ln 2 \cdot \text{Total Bytes}}$$
                - outputs: 모델 평가 척도 스칼라 `val_bpb`

            2. DCLM CORE Centered Accuracy 계산 ($(\text{Acc} - R) / (1 - R)$)
                - inputs: CORE ICL 태스크별 평가 정확도 $\text{Acc}$ , 무작위 기준선 정확도 $R$
                - algorithm: 모델의 순수 능력 향상도를 0(랜덤)부터 1(완벽)까지 정규화합니다:
                  $$\text{Centered} = \frac{\text{Acc} - R}{1 - R}$$
                  전체 세부 태스크의 중심화 정확도를 평균하여 단일 지표 $\text{CORE}$ 를 도출합니다.
                - outputs: 표준화된 베이스 모델 성능 지표 `core_metric`

            3. GSM8K Pass@k 추론 검증
                - inputs: GSM8K 테스트 질문 풀, 샘플링 개수 $k$
                - algorithm: $k$ 개의 응답 롤아웃 중 하나라도 수학적 정답과 일치하면 통과로 판정하는 패스율($\text{Pass@}k$)을 전 분산 랭크에 걸쳐 리듀스 집계합니다.
                - outputs: 단계별 $k=1 \dots 16$ 에 대한 `Pass@k` 정확도 리포트

        2. 고속 추론 엔진 (Inference Engine)
            1. Batch=1 Prompt Prefill (KV-Cache 초기화 및 Smear 은닉 상태 기록)
                - inputs: 사용자 프롬프트 토큰 ID 리스트 `tokens`
                - algorithm:
                    1. 단일 배치($B=1$)로 KVCache 객체를 할당하고 전체 프롬프트를 일괄 순방향 전파합니다.
                    2. 마지막 토큰의 정규화 임베딩을 `kv_cache.prev_embedding`에 저장하여 디코딩 시 단일 토큰 Smear 결합이 가능하도록 준비합니다.
                - outputs: 채워진 프리필 KVCache 및 첫 번째 생성 후보 로짓 텐서 `logits`

            2. Multi-sample KVCache 복제 (Prefill $\to$ Decode 구조)
                - inputs: 프리필 완료된 단일 캐시 `kv_cache_prefill` , 목표 샘플 수 `num_samples`
                - algorithm: 동일 프롬프트에 대한 병렬 디코딩을 위해, 프리필된 $K, V$ 텐서 슬라이스와 `prev_embedding` 상태를 `num_samples` 크기로 복제 복사(`prefill()`)하여 독립 디코딩 캐시를 구축합니다.
                - outputs: $B = \text{num\_samples}$ 크기의 복제된 디코딩 캐시 `kv_cache_decode`

            3. Flash Attention 3 In-place KV-Cache 슬라이딩 윈도우 업데이트
                - inputs: 신규 토큰 $q, k, v$ , KVCache 버퍼, `cache_seqlens`
                - algorithm: FA3 커널의 메모리 직접 접근을 통해 외부 할당 없이 $k, v$ 를 캐시 슬롯에 인플레이스 기록하고, 레이어별 윈도우 내 과거 토큰들에 대해서만 어텐션을 산출합니다.
                - outputs: 어텐션 출력 텐서 및 자동으로 $1$ 전진된 캐시 포인터

            4. Next-Token Top-k / Temperature 확률 샘플링
                - inputs: 마지막 위치 로짓 텐서 `logits` , 온도 $T$ , `top_k` , 난수 생성기 `rng`
                - algorithm:
                    1. $T=0$: `torch.argmax` 그리디 선택
                    2. $T > 0$: 상위 $k$ 개를 제외한 로짓을 $-\infty$ 로 차단한 후 Softmax 확률로부터 `torch.multinomial` 샘플링을 수행합니다.
                - outputs: 샘플링된 다음 정수 토큰 ID 배치

            5. Python REPL Tool Use 상태 머신
                1. <|python_start|> 트리거 기반 표현식 버퍼링
                    - inputs: 샘플링된 토큰 ID `next_token`
                    - algorithm: `<|python_start|>` 감지 시 `in_python_block = True` 플래그를 세우고, 이후 생성되는 토큰들을 문자열 표현식 버퍼에 수집합니다.
                    - outputs: 축적된 파이썬 수식 토큰 리스트 `python_expr_tokens`
                2. <|python_end|> 트리거 기반 파이썬 인터프리터 샌드박스 실행
                    - inputs: 닫는 태그 토큰 `<|python_end|>`
                    - algorithm: 버퍼링된 토큰을 문자열로 디코딩하고, 안전 샌드박스 함수(`use_calculator`)를 호출하여 타임아웃(3초) 제한 하에 안전하게 수식을 실행합니다.
                    - outputs: 인터프리터 실행 결과 문자열 (예: `"8"`)
                3. <|output_start|> {result} <|output_end|> 강제 토큰 큐(Forced Queue) 주입
                    - inputs: 실행 결과 문자열
                    - algorithm: 결과 문자열을 토큰화하여 `<|output_start|>` + 결과 토큰들 + `<|output_end|>` 시퀀스를 `forced_tokens` 데크(Deque)에 주입합니다. 다음 루프에서는 모델 샘플링 대신 이 큐에서 토큰을 강제 인출(Forced Injection)하며 마스크는 $0$ 을 부여합니다.
                    - outputs: 주입 대기 중인 토큰 큐 `forced_tokens`
                4. <|assistant_end|> / <|bos|> 감지 기반 생성 종료 제어
                    - inputs: 생성된 토큰 `next_token`
                    - algorithm: 어시스턴트 종료 태그 또는 문서 시작 태그가 생성되면 해당 배치의 `completed = True`로 전환하고, 모든 샘플이 완료되면 제너레이터 루프를 종료합니다.
                    - outputs: 최종 완결된 토큰 시퀀스 리스트 `results`
    - outputs:
        - 모델 압축률 및 벤치마크 정량 지표 리포트 (`val_bpb`, `core_metric`, `chatcore_metric`, `Pass@k`)
        - REPL 도구 실행 결과가 완전 통합되어 스트리밍 출력된 어시스턴트 답변 토큰 시퀀스 및 텍스트