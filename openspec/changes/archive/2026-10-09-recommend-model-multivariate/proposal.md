## Why

`recommend_model`(merlion/dashboard/models/model_select.py)은 학습 데이터에서 대상 변수 하나만 꺼내 후보를 검증한다. 그런데 대시보드는 사용자가 고른 다른 특징 변수(features)와 외생 변수(exog)를 함께 넣어 모델을 학습한다. 다변량 데이터에서는 이 때문에 "검증한 모델"과 "실제로 학습할 모델"이 달라지고, 고른 순위가 실제 학습 결과를 대변하지 못한다.

Merlion의 `ForecasterEnsemble` + `ModelSelector`는 이 점에서 다르다. 각 후보를 앙상블에 들어온 다변량 학습 데이터 그대로(`target_seq_index`로 대상 지정) 학습하고, 외생 변수를 `exog_data`로 넘겨 검증 구간의 예측에도 쓴다. 다변량 분석 관점에서는 이 "학습과 같은 조건의 검증"이 ModelSelector의 장점이므로 `recommend_model`에도 반영한다.

## What Changes

- `recommend_model`이 특징 변수와 외생 변수 목록을 받아, 후보를 대시보드 학습과 같은 열 구성과 외생 변수 처리로 검증한다.
  - 외생 변수를 지원하는 모델은 외생 변수를 `exog_data`로 받는다(검증 구간의 외생 값도 사용).
  - 지원하지 않는 모델은 대시보드 학습과 마찬가지로 외생 변수를 일반 열로 받는다.
- 후보별 추천 하이퍼파라미터를 계산할 때 특징 변수를 함께 넘긴다. 트리 모델과 VectorAR 추천기는 이미 특징 변수를 받도록 되어 있다.
- 학습 시간 추정과 학습 시간 상한 필터에 실제 변수 수를 반영한다.
- 특징 변수가 하나 이상이면 다변량 전용 모델인 VectorAR을 후보에 추가한다. 다변량 벤치마크에서 손해가 확인되면 추가하지 않는다.
- 단변량 입력(특징·외생 변수 없음)일 때 동작과 결과는 바뀌지 않는다.
- `ForecastModel.recommend_model`의 시그니처를 다음과 같이 확장한다. 기존 호출과 호환된다.
  ```python
  ForecastModel.recommend_model(train_df, target_column, horizon,
                                feature_columns=None, exog_columns=None, algorithms=None) -> ModelRecommendation
  ```
  이 시그니처는 병렬로 진행하는 `add-forecast-auto-train` 변경과의 계약이다.
- 다변량 벤치마크로 개선 여부를 확인한다. 데이터는 SKAB 8개 센서, `data/multivariate`, Walmart 외생 변수 데이터다.

## Capabilities

### New Capabilities
- `forecast-model-selection`: 학습 데이터로 예측 알고리즘과 하이퍼파라미터를 고르는 기능이다. 후보 구성, 검증 방식, 제외·실패 처리, 다변량·외생 변수 처리를 다룬다. 기존 `recommend_model` 동작은 아직 스펙이 없어 이번에 함께 문서화한다.

### Modified Capabilities
(없음)

## Impact

- 코드:
  - `merlion/dashboard/models/model_select.py`: 열 구성, 외생 변수 처리, 후보 목록
  - `merlion/dashboard/models/forecast.py`: `recommend_model` 시그니처, 그리고 학습과 검증이 함께 쓰는 열 배치 함수
- 테스트: `tests/dashboard/test_model_select.py`
- 벤치마크 스크립트는 scratchpad에 두고, 결과 요약은 커밋 메시지와 `model_select.py` docstring에 남긴다.
- 병렬 작업 `add-forecast-auto-train`은 위 시그니처만 사용한다. 이 변경은 `pages/`와 `callbacks/`를 수정하지 않는다.
- UI 연결은 `add-forecast-auto-train`의 Auto 모드가 맡는다.
  - Auto 모드는 "Select Other Features"와 "Select Exogenous Variables" 선택 상태로 단변량과 다변량을 판별한다.
  - 다변량이면 선택한 변수를 `feature_columns`와 `exog_columns`로 넘긴다. 이때 이 변경의 다변량 검증과 다변량 파라미터 추천이 적용된다.
  - Auto를 쓰지 않는 수동 학습은 `recommend_model`을 호출하지 않으므로 영향이 없다.
