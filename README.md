# 멀티모달 긴급도-감정 상태 판별 프로젝트

- **팀원**: 김민서(팀장), 김상윤(발표자), 이상준, 강지수, 김아람
- **기간**: 2025년 10월 17일(금) ~ 23일(목)
- **발표일**: 2025년 10월 24일 

---

## 1. 프로젝트 개요 (Introduction)

### 1.1. 프로젝트 목표

- 119 신고 음성 파일(`.wav`)과 텍스트 데이터(`.json`) 활용
- **Multi-modal** (음성+텍스트) 및 **Multi-task** (긴급도+감정) 모델을 구현하여, 접수된 신고의 **긴급도(urgencyLevel)**와 **감정 상태(sentiment)** 판별.
- AI 실무 프로세스 경험 및 딥러닝 모델 설계/적용 능력 습득을 목표.

### 1.2. 팀원 및 역할

- `김민서` : EDA, 데이터 정제 코드 작성, 데이터 전처리, HuBERT 융합 실험, default 모델 작성, 문서 작성
- `김성윤` : 베이스라인 작성 및 실험, 모델 구조 변화 실험, fine-tuning 실험, 발표 준비, 문서 작성
- `이상준` : EDA, 데이터 정제, 베이스라인 모델 튜닝, default 모델 실험, 문서 작성
- `강지수` : EDA, Robert 실험 및 융합 실험, 문서 작성
- `김아람` : EDA, 데이터 샘플링 코드 작성, task 변화 실험, fine-tuning 융합 실험, 문서 작성

---

## 2. 문제 정의 (Problem Definition)

- **주어진 과제**: 119 신고 음성과 텍스트를 입력받아, 해당 신고의 긴급도와 발화자의 감정을 동시에 분류.
- **핵심 도전 과제 (Challenge)**:
  1.  **Multi-modal**: 음성(Audio)과 텍스트(Text)라는 두 가지 다른 형태의 데이터를 어떻게 효과적으로 융합(fusion)할 것인가?
  2.  **Multi-task**: '긴급도'와 '감정'이라는 두 가지 다른 목표(task)를 하나의 모델로 동시에 학습시킬 것인가?
- **제한 사항**: 음성 파일(.wav)과 발화된 음성을 변환한 텍스트를 사용해 긴급도(urgencyLevel)와 감정 상태(sentiment)를 판별하는 멀티 모달리티, 멀티 태스크 모델을 구현.
- **원본 데이터**: [위급상황 음성/음향 (고도화) - 119 지능형 신고접수 음성 인식 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71768)

    <img src="./image/data_aihub.png" width="400"/>

---

## 3. 데이터 분석 및 전처리 (Data E&D & Preprocessing)

### 3.1. 사용 데이터 선정

- **데이터 출처**: AI Hub - 119 지능형 신고접수 음성 인식 데이터
- **데이터 선별**:
  - 원본 데이터(서울, 광주, 인천)의 전체 용량이 방대하여, **인천 데이터(총 `22,797`건)** 만 선별하여 사용하기로 결정.
- **학습 전략**:
  - 전체 `22,797`건 역시 초기 모델링에 많다고 판단.
  - **1단계 (모델 검증)**: `500`개의 소규모 데이터로 훈련을 우선 시도하여 모델 파이프라인 검증.
  - **2단계 (최종 학습)**: 이후 `2,000`개의 데이터를 사용하여 최종 모델 학습 진행.

### 3.2. 데이터 구조 (JSON 예시)

- 데이터는 `.wav` 오디오 파일과 매칭되는 `.json` 메타데이터로 구성.

#### 3.2.1. 원천데이터

  <img src="./image/data_wav.png" />

#### 3.2.2. 라벨데이터

```json
{
  "_id": "64d9fdff3e12da15ae3a359e",
  "audioPath": "20230814/Incheon/2023/02/07/016/converted_20230207065612_4016-016.wav",
  "recordId": "9d7cc435cca747a1a731",
  "status": 12,
  "startAt": 0,
  "endAt": 94200,
  "utterances": [
    {
      "id": "fc2db008",
      "startAt": 25173,
      "endAt": 29406,
      "text": "부계동 어, 부계역 바로 앞에 있는 대동아파트거든요. ",
      "speaker": 0
    },
    {
      "id": "wavesurfer_otjm8pn3rq",
      "startAt": 40433,
      "endAt": 46847,
      "text": "아들 분 아드님이 어깨 탈골 돼서, 지금 그 아드님이랑 같이 있는 분 뭐, 연락처 저, 있나요?",
      "speaker": 1
    }
  ],
  "mediaType": "mobile",
  "gender": "M",
  "address": "인천광역시 부평구 부개동",
  "disasterLarge": "구급",
  "disasterMedium": "질병(중증 외)",
  "urgencyLevel": "중",
  "sentiment": "불안/걱정",
  "symptom": ["기타통증"],
  "triage": "준응급증상"
}
```

### 3.3. 주요 Feature 및 Target Label

<img src="./image/main_feature.jpg" width="500" />

- **Input Features (Multi-modal)**
  - **Audio**: `audioPath`를 이용해 로드한 `.wav` 파일
  - **Text**: `utterances` 내의 `text` 필드
- **Target Labels (Multi-task)**
  - **긴급도**: `urgencyLevel` ("상", "중", "하")
  - **감정**: `sentiment` ("불안/걱정", "당황/난처", "중립", "기타부정")

### 3.4. 데이터 탐색 (EDA - Exploratory Data Analysis)
- 전체 데이터셋 규모가 약 22,000건으로 크고,
  초기 단계에서 모든 데이터를 정밀 분석하기에는 시간적 제약이 존재하였다.
- 이에 따라 EDA 단계에서는 인천 데이터 중 `disasterLarge` 기준으로
  각 카테고리별 50건씩, 총 200건을 샘플링하여
  데이터 구조 이해 및 feature–label 간 관계 탐색에 집중하였다.
- 본 EDA 결과는 통계적 결론보다는
  이후 모델 설계 및 feature 선택을 위한 정성적 근거로 활용되었다.


#### 3.4.1. **음성 데이터 분석**
- MFCC의 평균, 표준편차, 발화 속도 분포를 통해
  감정 상태별 음성 신호의 차이를 정성적으로 확인하고자 하였다.
- 일부 감정 클래스에서 MFCC 통계량의 분포 차이가 관찰되었으나,
  단순 통계 기반 특징만으로는
  긴급도 및 복합적인 감정 패턴을 명확히 구분하기에는 한계가 있었다.
- 이를 통해 전통적인 음성 특징(MFCC)보다는
  Wav2Vec2, HuBERT와 같은 end-to-end 음성 표현 학습 모델의 필요성을 확인하였다.

|                                             |                                         |
| ------------------------------------------- | --------------------------------------- |
| <img src="./image/mfcc_mean.png" />         | <img src="./image/mfcc_std.png" />      |
| <img src="./image/mfcc_speech_speed.png" /> | <img src="./image/mfcc_mean_std.png" /> |

#### 3.4.2. **메타 데이터 분석**
- 통화 길이(endAt - startAt) 분포를 분석한 결과, 다수의 샘플은 짧은 길이에 집중되어 있으나 일부 매우 긴 통화로 인한 long-tail 분포가 존재함을 확인하였다.
- 긴급도가 높은 신고일수록 평균 통화 길이가 길어지는 경향이 관찰되었으며, 불안/걱정, 당황/난처와 같은 감정 상태에서도 중립 대비 상대적으로 긴 발화 길이를 보였다.
- 이는 발화 길이가 긴급도 및 감정 상태를 반영하는 보조적인 신호로 활용될 수 있음을 시사한다.
- 지역(구/시) 단위 분석 결과, 신고 건수는 특정 지역에 집중되는 경향을 보였으나 평균 긴급도는 지역 간 큰 차이를 보이지 않았다.
- 이에 따라 지역 정보는 모델 성능 개선보다는 데이터 분포 이해 및 편향 분석 목적에 더 적합한 feature로 판단하였다.


|                                                                   |                                                          |
| ----------------------------------------------------------------- | -------------------------------------------------------- |
| <img src="./image/txt_urgencyLevel.png" />                        | <img src="./image/txt_sentiment.png" />                  |
| <img src="./image/txt_call_Duration_by_urgency.png" />            | <img src="./image/txt_call_duration_by_sentiment.png" /> |
| <img src="./image/txt_triage.png" />                              | <img src="./image/txt_sympton.png" />                    |
| <img src="./image/txt_sympton_urgencylevel.png" />                | <img src="./image/txt_sympton_sentiment.png" />          |
| <img src="./image/txt_단위신고건수_heatmap.png" />                | <img src="./image/txt_시구단위평균_heatmap.png" />       |
| <img src="./image/txt_urgencylevel_sentiment_행정규화비율.png" /> | <img src="./image/txt_신고자_수보자.png" />              |
| <img src="./image/txt_endat_distribution_outlier.png" />          | <img src="./image/txt_상관관계_heatmap.png" />           |

### 3.5. 모델 입력을 위한 전처리 (Feature Preprocessing)

- 원본 `.json`과 `.wav` 파일은 사용하기 복잡하므로, 모델 학습에 필요한 정보만 추출하여 훈련용 `train.csv` 와 `train_audio.tar` , 테스트용 `validation.csv` 와 `validation_audio.tar`파일로 통합.

<img src="./image/data_refinement.jpg" alt="데이터 정제 파이프라인" width="700"/>

#### 3.5.1. 오디오 처리 (Audio Processing)

- **입력:** 원본 `.JSON` 파일과 원본 `.WAV` 오디오 파일
- **프로세스:**
  - `.JSON` 파일에 포함된 `utterances` 정보, 특히 **`startAt`** (시작 시간)과 **`endAt`** (종료 시간) 타임스탬프 정보를 읽어와서
  - 이 타임스탬프를 기준으로, 원본 `.WAV` 파일에서 해당하는 오디오 구간을 잘라내고(slicing).
  - 하나의 원본 파일에서 여러 개의 분할된 `.WAV` 조각 파일들이 생성됨.
- **출력:** 분할된 모든 `.WAV` 조각 파일들을 하나의 **`.tar`** 아카이브 파일(예: `train_audio.tar`)로 통합.

#### 3.5.2 텍스트 및 메타데이터 처리 (Text & Metadata Processing)

- **입력:** 원본 `.JSON` 파일
- **프로세스**:
  - **텍스트(Text) 추출:** `utterances` 배열(list) 안에 나뉘어 저장된 여러 텍스트 조각들을 모두 추출하여 하나의 단일 텍스트 문자열로 결합합니다.
  - **메타데이터(Meta data) 추출:** 모델 학습에 필요한 다음 항목들을 `.JSON` 파일에서 직접 추출.
    - `gender`, `address`, ` disaster` , `disasterLarge` , `urgencyLevel`, `sentiment`, `symptom`, `triage`
- **출력:** 추출된 단일 텍스트와 모든 메타데이터 항목들을 컬럼(column)으로 하는 **`.csv`** 파일(예: `train.csv`)을 생성.

#### 3.5.3. 최종 통합

- 생성된 `.csv` 파일에는 각 행(데이터 샘플)이 `.tar` 아카이브 내의 어떤 오디오 조각 파일과 매칭되는지를 알려주는 **`audio_path`** (오디오 경로) 컬럼이 존재 → 이를 통해 텍스트, 메타데이터, 오디오가 하나의 세트로 연결.

### 3.6. 데이터 처리 제약 및 엔지니어링 이슈
- 본 데이터셋은 음성 파일의 개별 용량이 크고 전체 규모 또한 방대하여, 원본 상태에서의 압축 해제(unzip) 및 오디오 분절(slicing) 과정 자체가 로컬 및 원격 학습 환경에서 상당한 부담으로 작용하였다.
- 초기에는 분절된 오디오 파일들을 개별 파일 경로로 참조하여 학습을 시도하였으나, 다수의 파일을 동시에 로드하는 과정에서 스토리지 I/O 병목으로 인해 학습 인스턴스의 디스크 자원이 한계에 도달하는 문제가 발생하였다.
- 이러한 문제를 해결하기 위해, 분절된 오디오 파일들을 개별 파일로 관리하는 대신 모든 오디오 조각을 하나의 `.tar` 아카이브 파일로 통합하고, 학습 시점에 필요한 오디오만을 코드 레벨에서 로드하여 메모리에 적재하는 방식으로 데이터 로딩 파이프라인을 변경하였다.
- 다만 전체 데이터셋에 대해 해당 방식을 적용하는 것은 압축 및 로딩 비용 측면에서 여전히 현실적인 제약이 존재하였기 때문에, 우선적으로 소규모 샘플 데이터를 사용하여 모델 구조와 학습 파이프라인이 정상적으로 동작하는 baseline 환경을 구축하는 것을 목표로 하였다.
- 그 결과, 모델 구조는 유지한 상태에서 데이터 수만 제한하여 안정적인 end-to-end 학습이 가능한 baseline을 성공적으로 구축하였으며, 이후 실험은 해당 파이프라인을 기반으로 단계적으로 확장하였다.
- 추가적으로 오디오 아카이브 포맷과 관련하여, 실제 파일은 `.tar.gz` 형식이었으나 파일 경로를 `.tar`로 지정하고 `tarfile` 로딩 시 `r:*` 옵션을 사용하면
  별도의 포맷 분기 처리 없이 정상적으로 로드가 가능함을 확인하였다.

---

## 4. 베이스 모델 아키텍처 (Model Architecture)

- 텍스트, 음성, 메타데이터를 입력받는 Multi-modal, Multi-task (Urgency:회귀, Sentiment:분류) 모델.
- Text Encoder: KcELECTRA
- Audio Encoder: Wav2Vec2
- Fusion: 텍스트(txt_emb)와 오디오(audio_emb) 임베딩을 Sum(+).

### 4.1. 초기 모델 구상

<img src="./image/base1.png" />

### 4.2. Baseline 구조

<img src="./image/base2.jpg" />

```python
class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_name, meta_maps: Dict[str, Dict],
                 audio_emb_dim=256, joint_dim=256, num_classes_sentiment=4,
                 audio_input_dim=None):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, joint_dim)

        if audio_input_dim is None:
            audio_input_dim = w2v_model.config.hidden_size  # wav2vec2-XLSR의 hidden dim

        self.audio_proj_shared   = nn.Linear(audio_input_dim, audio_emb_dim)
        self.audio_proj_urgency  = nn.Linear(audio_emb_dim, joint_dim)
        self.audio_proj_sentiment= nn.Linear(audio_emb_dim, joint_dim)

        self.meta_embs = nn.ModuleDict()
        for k, mapping in meta_maps.items():
            num_embeddings = len(mapping) + 1
            emb_dim = min(8, num_embeddings)
            self.meta_embs[k] = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim, padding_idx=0)
        self.meta_out_dim = sum([emb.embedding_dim for emb in self.meta_embs.values()]) if self.meta_embs else 0

        self.urgency_head = nn.Linear(joint_dim + self.meta_out_dim, 1)
        self.sentiment_head = nn.Sequential(
            nn.Linear(joint_dim + self.meta_out_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes_sentiment)
        )

    def forward(self, input_ids, attention_mask, audio_feat, meta_idx):
        B = input_ids.size(0)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_emb = self.text_proj(txt_out.last_hidden_state[:, 0, :])

        # audio_feat: (B, D, T)
        audio_mean = audio_feat.mean(dim=2)  # (B, D)
        audio_emb = self.audio_proj_shared(audio_mean)
        audio_emb_urgency  = self.audio_proj_urgency(audio_emb)
        audio_emb_sentiment= self.audio_proj_sentiment(audio_emb)

        if not self.meta_embs:
            meta_emb = torch.zeros(B, 0, device=input_ids.device)
        else:
            if meta_idx.dim() == 1:
                meta_idx = meta_idx.unsqueeze(0).expand(B, -1)
            meta_embeddings = [self.meta_embs[k](meta_idx[:, i].clamp(0, self.meta_embs[k].num_embeddings-1))
                               for i, k in enumerate(self.meta_embs.keys())]
            meta_emb = torch.cat(meta_embeddings, dim=1)

        joint_urgency   = txt_emb + audio_emb_urgency
        joint_sentiment = txt_emb + audio_emb_sentiment

        urg_out  = self.urgency_head(torch.cat([joint_urgency, meta_emb], dim=1)).squeeze(1)
        sent_out = self.sentiment_head(torch.cat([joint_sentiment, meta_emb], dim=1))
        return urg_out, sent_out
```

- **사용한 모델 (Text)**: beomi/KcELECTRA-base
  - 선정 이유: 한국어 댓글 등 구어체 데이터로 학습되어, 119 신고 STT 텍스트 처리에 적합할 것으로 판단.
- **사용한 모델 (Audio)**: facebook/wav2vec2-xls-r-300m
  - 선정 이유: 다국어 음성으로 사전 학습되어, 별도 fine-tuning 없이도 음성 특징(prosody, tone) 추출에 효과적.

### 4.3. Audio Encoder 선정 배경 (MFCC vs Wav2Vec2)

- **초기 시도 (MFCC)**: 
  - 음성 신호 처리에서 전통적으로 사용되는 MFCC(Mel-frequency cepstral coefficients) 를 사용하여 베이스라인을 구축하려 시도.
  - **한계점**: MFCC는 고정된 특징(hand-crafted feature)으로, 긴급 신고 음성에 담긴 미세한 **운율(prosody)**이나 **감정적 뉘앙스**를 충분히 포착하지 못함. 또한, Text Embedding과의 차원 불일치 문제를 해결하기 위해 추가적인 복잡한 레이어 설계가 필요했음.
  
- **최종 결정 (Wav2Vec2)**: 
  - **선정 이유**: 대량의 음성 데이터로 사전 학습된 Wav2Vec2는 음성의 문맥적 정보(Contextual Information)를 스스로 학습한 모델임.
  - **이점**: 별도의 복잡한 전처리 없이 raw waveform을 입력받아 고수준의 특징을 추출할 수 있으며, Transformer 기반 구조 덕분에 Text Encoder(KcELECTRA)와의 융합(Fusion)이 구조적으로 용이함.
 
### 4.4. 모델 후보 항목 및 특징

본 프로젝트에서는 멀티모달(음성+텍스트) 및 멀티태스크(긴급도+감정) 문제 특성을 고려하여,
각 모달리티별로 여러 사전학습(pretrained) 모델을 후보군으로 선정하고 비교하였다.

#### 4.4.1. Audio Encoder 후보

##### 1) MFCC (Mel-Frequency Cepstral Coefficients)
- **특징**
  - 음성을 주파수 영역으로 변환한 뒤 Mel-scale 필터를 적용하여 계수 형태로 표현하는 전통적 음성 특징
  - 계산 비용이 낮고 해석 가능성이 높음
- **장점**
  - 빠른 전처리 및 경량 모델 구성 가능
  - 소규모 데이터 환경에서 안정적인 특징 제공
- **한계**
  - Hand-crafted feature로 감정, 억양, 운율(prosody) 등 고수준 음성 정보를 충분히 반영하기 어려움
  - 2D 특징 맵 형태로 생성되어 Transformer 기반 텍스트 임베딩과의 융합이 까다로움
- **채택 여부**
  - ❌ 초기 베이스라인 시도 후 성능 및 융합 한계로 최종 모델에서는 미채택


##### 2) Wav2Vec 2.0
- **특징**
  - Raw waveform을 입력으로 받아 자기지도 학습(Self-supervised learning)을 통해 음성 표현을 학습하는 모델
  - CNN 기반 feature encoder + Transformer 구조
- **장점**
  - 라벨이 없는 대규모 음성 데이터로부터 고수준 음성 표현 학습 가능
  - 감정, 억양, 발화 스타일 등 문맥적 음성 정보(Contextual information) 포착에 강점
  - Transformer 기반 구조로 텍스트 인코더와의 융합이 용이
- **채택 여부**
  - ✅ 최종 Audio Encoder로 채택

##### 3) HuBERT (Hidden-Unit BERT)
- **특징**
  - 음성 프레임을 클러스터링하여 pseudo-label을 생성하고, 이를 예측하는 방식으로 학습
  - 연속적인 음성 신호를 이산적인 단위(hidden unit) 예측 문제로 변환
- **장점**
  - Wav2Vec2 대비 학습 안정성이 높음
  - 음성의 구조적·음소 단위 정보를 잘 반영
- **채택 여부**
  - 🔁 실험용 대안 모델로 사용 (융합 실험 수행)

#### 4.4.2. Text Encoder 후보

##### 1) KcELECTRA
- **특징**
  - ELECTRA 구조 기반의 한국어 특화 사전학습 모델
  - 한국어 댓글, SNS, 구어체 데이터 중심 학습
- **장점**
  - 비표준 표현, 감정 표현, 구어체 문장 처리에 강점
  - Replaced Token Detection 방식으로 학습 효율이 높음
- **채택 여부**
  - ✅ 베이스라인 및 최종 모델의 Text Encoder로 채택


##### 2) RoBERTa (klue/roberta-base)
- **특징**
  - BERT 구조를 기반으로 NSP 제거, 대규모 데이터 및 장시간 학습을 통해 성능을 향상시킨 모델
  - KLUE 코퍼스로 학습된 한국어 특화 버전
- **장점**
  - 문맥 이해 능력이 뛰어나 복잡한 문장 구조와 감정 표현에 강점
  - 감정 분류 태스크에서 우수한 성능
- **채택 여부**
  - 🔁 Text Encoder 대체 실험에 사용


#### 4.4.3. 모델 후보 요약

| Modality | Model        | 핵심 특징 요약                                  | 사용 여부 |
|----------|--------------|-----------------------------------------------|-----------|
| Audio    | MFCC         | 전통적 주파수 기반 특징, 해석 용이             | ❌        |
| Audio    | Wav2Vec2     | Self-supervised, 문맥적 음성 표현 학습         | ✅        |
| Audio    | HuBERT       | Clustering 기반 pseudo-label 예측              | 🔁        |
| Text     | KcELECTRA    | 한국어 구어체 특화, 학습 효율 우수              | ✅        |
| Text     | RoBERTa      | 문맥 표현력 강화된 BERT 계열                   | 🔁        |

---

## 5. 실험/학습

```
<!-- 실험 조건 -->
TARGET_N = 500
NUM_EPOCHS = 10
```

### 5.0. baseline

- Text: KcELECTRA, Audio: Wav2Vec2, Fusion: Sum
- 학습 그래프

    <img src="./image/exp_baseline.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.7474
  Test MSE (urgencyLevel): 0.7373
  Test Accuracy (sentiment): 0.5949
  Test Weighted F1 (sentiment): 0.4468
  ```

### 5.1. klue/roberta

- Text: klue/roberta-base로 변경
- 학습 그래프

    <img src="./image/exp_klue_roberta.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.7884
  Test MSE (urgencyLevel): 0.8147
  Test Accuracy (sentiment): 0.5879
  Test Weighted F1 (sentiment): 0.4733
  ```

### 5.2. HuBERT

- Audio: HuBERT로 변경
- 학습 그래프

    <img src="./image/exp_hubert.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.7780
  Test MSE (urgencyLevel): 0.9159
  Test Accuracy (sentiment): 0.5837
  Test Weighted F1 (sentiment): 0.4706
  ```

### 5.3. 속성(attribute) 조절

- Meta-data 임베딩 실험
- 학습 그래프

    <img src="./image/exp_attribute.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.7443
  Test MSE (urgencyLevel): 0.7166
  Test Accuracy (sentiment): 0.5952
  Test Weighted F1 (sentiment): 0.4489
  ```

### 5.4. model 튜닝

- 학습 그래프

    <img src="./image/exp_model_tunning.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.8117
  Test MSE (urgencyLevel): 0.8543
  Test Accuracy (sentiment): 0.5795
  Test Weighted F1 (sentiment): 0.4732
  ```

### 5.5. Concat

- Fusion: Concat으로 변경
- 학습 그래프

    <img src="./image/exp_concat.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.7283
  Test MSE (urgencyLevel): 0.7616
  Test Accuracy (sentiment): 0.6075
  Test Weighted F1 (sentiment): 0.5902
  ```

### 5.6. fine-tuning 일부

- Encoder 일부 Layer Fine-tuning
- 학습 그래프

    <img src="./image/exp_fine_tunning.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.7939
  Test MSE (urgencyLevel): 0.8275
  Test Accuracy (sentiment): 0.5959
  Test Weighted F1 (sentiment): 0.4467
  ```

### 5.7. 회귀to분류

- Urgency Task: Regression -> Classification 변경
- 학습 그래프

    <img src="./image/exp_회귀to분류.png" />

- 테스트 결과

  ```
  Test Weighted F1 (urgency): 0.4195
  Test Accuracy (sentiment): 0.6215
  Test Weighted F1 (sentiment): 0.6012
  ```

### 5.8. concat + model 튜닝 + HuBERT

- 학습 그래프

    <img src="./image/exp_concat_model_hubert.png" />

- 테스트 결과

  ```
  Test MAE (urgencyLevel): 0.9552
  Test MSE (urgencyLevel): 1.1187
  Test Accuracy (sentiment): 0.5756
  Test Weighted F1 (sentiment): 0.5090
  ```

---

## 6. 융합 실험

<img src="./image/list.jpg" />

## 6.1. Default ( classification, features, concat)

- 기존 base 모델에서 아래 항목을 수정하여 default 모델을 제작
- classification : 기존 모델에서 urgencyLevel을 regression에서 classification으로 수정
- features : ‘gender’, ‘disasterLarge’, ‘disasterMedium’ 만을 메타데이터 features로 반영
- concat : 오디오와 텍스트 context vector를 sum 에서 concat하는 방식 채택

- 학습 그래프

  <img src="./image/md_default.png" />

- 테스트 결과

  ```
  Test Accuracy (urgency): 0.3901
  Test Weighted F1 (urgency): 0.3942
  Test Accuracy (sentiment): 0.5823
  Test Weighted F1 (sentiment): 0.5273
  ```

## 6.2. [default + roberta]

**모델 : klue/roberta-base**

- 특징 : 한국어 자연어 처리에 최적화된 RoBERTa 기반 사전학습 모델, 대규모 한국어 데이터셋(KLUE Corpus)으로 학습되어 다양한 문맥 이해와 감정·의도 인식에 강점
- 선정 이유 : 한국어 긴급통화와 같이 복잡한 문맥과 감정이 혼합된 데이터를 처리하기 위해, 문맥 표현력이 우수하고 한국어 문장 구조를 잘 반영하는 klue/roberta-base을 선택

- 학습 그래프

  <img src="./image/md_default_roberta.png" />

- 테스트 결과

  ```
  Test Accuracy (urgency): 0.4230
  Test Weighted F1 (urgency): 0.2701
  Test Accuracy (sentiment): 0.6502
  Test Weighted F1 (sentiment): 0.5995
  ```

## 6.3. [default + hubert]

**모델 : facebook/hubert-base-ls960**

- 특징 : 음성 신호를 비지도 학습으로 전처리된 음소 단위(hidden unit)로 표현, 발화의 내용뿐 아니라 화자의 억양, 감정, 음성적 특징까지 효과적으로 인코딩하는 모델
- 선정 이유 : 긴급 통화 음성 데이터는 잡음, 억양, 감정 표현 등이 다양하기 때문에, 단순한 스펙트럼 특징보다 고수준 음성 표현을 학습할 수 있는 HuBERT 모델을 사용하여 감정 및 긴급도 분류의 정확도를 향상시키기 위해 선정

- 학습 그래프

  <img src="./image/md_default_hubert.png" />

- 테스트 결과

  ```
  Test Accuracy (urgency): 0.3995
  Test Weighted F1 (urgency): 0.3993
  Test Accuracy (sentiment): 0.5221
  Test Weighted F1 (sentiment): 0.5140
  ```

## 6.4. [default + 파인튜닝]

- 파인튜닝 실험에서는 **모델의 상위 2개 레이어의 가중치 학습을 허용**하여, 고수준 언어·음성 표현을 미세 조정(fine-tuning)함으로써 **도메인 적응력과 성능 향상**을 기대함.
- 기존 실험에서는 **모든 pretrained 모델의 가중치를 freeze**하고, **MLP 헤드만 학습**하도록 설정함.

- 학습 그래프
  <img src="./image/md_default_finetunning.png" />

- 테스트 결과
  ```
  Test Accuracy (urgency): 0.4317
  Test Weighted F1 (urgency): 0.4164
  Test Accuracy (sentiment): 0.6001
  Test Weighted F1 (sentiment): 0.5885
  ```

---

## 7. 프로젝트 일지

- 10/17 금
  - 일정 수립, EDA 역할 정하기
- 10/20 월
  - 전처리 한 뒤 프로토타입 만들기
  - 모델링 세분화 / 전처리 세분화 팀 나눠서 조사
- 10/21 화
  - 모델&전처리 탐색, 오디오 데이터 압축 테스크
    - 데이터 명이나 특징 기반 프로젝트 분석해놓은 블로그 참고
    - 전처리 기법 → 대회 참고
  - baseline 코드로 테스트 환경 구성, 테스트 학습
- 10/22 수
  - 텍스트 모델 탐색, 실험 및 테스트
- 10/23 목
  - 추가 융합 실험 진행
  - 프로젝트 문서 작성

---

## 8. 결론 및 해석

### 8.1 결론

- **최고 성능 모델**: Concat 방식이 감정 분류에서 가장 우수 (Accuracy 0.6075, F1 0.5902)
- **긴급도 예측**: 전반적으로 MAE 0.72~0.95 수준으로 난이도 높음 (회귀 문제의 한계)
- **융합 실험**: roberta 적용 시 감정 분류 성능 향상 (Accuracy 0.6502, F1 0.5995)
- **파인튜닝 효과**: 상위 2개 레이어 파인튜닝으로 긴급도(0.4317), 감정(0.6001) 모두 개선
- **모달리티 융합**: Sum 방식보다 Concat 방식이 텍스트-오디오 정보 통합에 효과적

### 8.2 해석

**1. Concat > Sum: 모달리티 융합 전략의 결정적 차이**
- Sum은 서로 다른 특성(텍스트 의미 vs 음성 감정)을 평균화하여 고유 정보 손실
- Concat은 각 모달리티의 독립적 특징을 보존하여 모델이 선택적으로 학습 가능
- 실험 결과: Concat 방식이 감정 F1 0.59로 baseline 0.45 대비 32% 향상

2. **부분 Fine-tuning의 효율성**
- 상위 2개 레이어만 학습해도 urgency 0.43, sentiment 0.60으로 개선

3. **데이터 규모보다 파이프라인 검증 우선**
- 완벽한 모델보다 작동하는 end-to-end 시스템 구축이 실전 프로젝트의 핵심
- 1주일 제약에서 다양한 실험 병렬 수행으로 최적 조합 발견 (빠른 iteration)

### 8.3. 실험 중 시행착오 및 학습 불안정성 분석

본 프로젝트에서는 다양한 모델 구조와 feature 구성을 실험하는 과정에서,
일부 설정에서 과적합(overfitting) 및 학습 불안정성 문제가 반복적으로 관찰되었다.
본 절에서는 이러한 현상이 발생한 원인과 그에 따른 설계 변경 과정을 정리한다.


#### 8.3.1. HuBERT 모델의 Task별 성향 차이

- HuBERT 기반 Audio Encoder는
  **Sentiment Classification 태스크에서는 비교적 안정적인 성능**을 보였으나,
  **Urgency Regression 태스크에서는 과적합 및 성능 변동성이 크게 나타났다.**
- 이는 HuBERT가 음성의 억양, 감정, 발화 스타일과 같은
  분류에 유리한 고수준 음성 표현을 잘 학습하는 반면,
  연속적인 수치 예측에는 상대적으로 불안정한 특성을 보이기 때문으로 해석된다.


#### 8.3.2. Meta Feature 구성에 따른 Overfitting 문제

- 초기 실험에서는 `triage`, `symptom` 등 다양한 메타데이터를 함께 사용하였으나,
  훈련 성능 대비 검증 성능이 급격히 저하되는 과적합 현상이 발생하였다.
- 이후 해당 feature들을 제거하고,
  상대적으로 정보량이 적은 **`gender` (대/소 분류 수준)** 만을 유지한 결과,
  일부 데이터 샘플링 기반 검증 실험에서 과적합이 완화되는 경향을 확인하였다.
- 다만, 과적합은 완화되었으나
  **F1 score가 학습 과정에서 정체되는 현상**은 지속되었으며,
  이는 데이터 규모, 클래스 불균형,
  또는 멀티태스크 학습 간 gradient 간섭 문제로 추정된다.


#### 8.3.3. Urgency Task 설계 변경의 영향 (Regression → Classification)

- Urgency 예측을 회귀 문제로 설정했을 때,
  전반적으로 학습이 불안정하고 예측 분산이 큰 경향을 보였다.
- Classification 방식으로 전환한 이후에도
  긴급도 예측은 여전히 변동성이 존재했으나,
  **Sentiment Classification 성능은 비교적 일관되게 유지되거나 개선**되었다.
- 이는 긴급도 라벨이 상대적으로 주관적이고 경계가 모호한 반면,
  감정 라벨은 음성 및 텍스트 양쪽에서 비교적 명확한 신호를 가지기 때문으로 해석된다.


---

## 9. 프로젝트 중 이슈 및 향후 개선 방향

### 9.1. 대용량 오디오 데이터 처리 및 파이프라인 구축 (Engineering Challenge)
초기 학습 파이프라인 구축 단계에서 2만 건이 넘는 고용량 오디오 데이터를 다루는 데 큰 엔지니어링적 난관이 있었습니다.

- **I/O 병목 및 로딩 속도 이슈**:
  - **[Issue]** 오디오 원본 파일(.wav)의 총용량이 너무 커서, Colab 및 로컬 환경에서 개별 파일을 로드할 때 I/O 병목 현상 발생. 단순 데이터 로딩에만 수 시간이 소요되어 학습 사이클(Iteration)을 빠르게 돌리기 어려웠음.
  - **[Solution]** 데이터를 개별 파일로 두지 않고 `.tar` 아카이브 형태로 묶어 시퀀셜하게 읽어들이는 방식을 도입하여 I/O 오버헤드를 획기적으로 줄임.
- **데이터 전처리 및 관리의 어려움**:
  - 데이터 탑재(Mount), 로딩, 그리고 타임스탬프 기반 분절(Slicing) 작업의 복잡성.
  - 대용량 압축 파일(.zip, .tar, .tar.gz)의 압축 해제 시 용량 부족 및 시간 소요 문제.

### 9.2. 모델 아키텍처 및 Feature Engineering (Modeling Challenge)
음성 데이터의 특성을 텍스트와 효과적으로 결합하기 위해 Feature Extractor를 선정하는 과정에서 시행착오를 겪었습니다.

- **MFCC의 한계와 Wav2Vec2 도입**:
  - **[Issue - MFCC]** 초기 베이스라인에서는 전통적인 **MFCC** 방식을 시도했으나, 2D 이미지 형태의 특징 맵이 생성되어 1D 텍스트 임베딩과 차원(`Dimension Mismatch`)을 맞추기 까다로웠음. 또한, 핸드 크래프트(Hand-crafted) 특징인 MFCC로는 긴급 신고 음성의 미세한 감정과 운율 정보를 담기에 역부족이라 판단함.
  - **[Solution - Wav2Vec2]** 이를 해결하기 위해 Transformer 기반의 **Wav2Vec2** 모델을 도입. Raw Waveform에서 문맥 정보(Contextual Info)가 포함된 고수준의 특징을 추출하고, 텍스트 인코더(KcELECTRA)와 유사한 차원 구조를 가져 융합(Fusion) 성능을 개선함.

### 9.3. 학습 환경 및 데이터 품질 (Environment & Data Quality)
- **GPU 메모리 이슈 (OOM)**:
  - Pretrained 모델(Wav2Vec2 + RoBERTa)을 동시에 로드하고 Fine-tuning(상위 레이어 해제)을 시도하는 과정에서 GPU 메모리 한계(Out Of Memory) 빈번 발생. → 배치 사이즈 조절 및 Gradient Accumulation 등으로 대응.
- **결측치(Missing Values) 이슈**:
  - 일부 데이터의 누락으로 인해 Loss가 `NaN`으로 뜨며 학습이 불안정해지는 현상 발생 → 전처리 단계에서 결측 데이터 필터링 강화.
- **클라우드 환경 제약**:
  - 잦은 데이터 요청으로 인한 구글 드라이브 API Block(접근 제한) 현상으로 학습 중단 위기.

### 9.4. 프로젝트 제약 사항 (Constraints)
- **짧은 프로젝트 기간 (1주일)**:
  - Multi-modal & Multi-task라는 복잡한 주제를 1주일 내에 소화해야 했기에, 더 다양한 최신 모델(SOTA)이나 복잡한 앙상블 기법을 시도해보지 못한 아쉬움이 남음.
  - 하지만 제한된 시간 내에 **'데이터 전처리 - 모델링 - 검증'**의 전체 파이프라인을 완주했다는 점에 의의를 둠.

