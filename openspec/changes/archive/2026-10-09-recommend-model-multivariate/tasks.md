## 1. 열 배치 공유

- [x] 1.1 `ForecastModel.train`에서 열 정규화·배치·외생 변수 분리 로직을 함수로 분리하고 `train`이 이를 쓰게 바꾼다. 검증: `python -m pytest tests/dashboard/test_forecast_model.py -q` 통과
- [x] 1.2 분리한 함수의 단위 테스트를 추가한다. 외생 지원 모델은 exog가 분리되고, 비지원 모델은 열로 남으며, `target_seq_index`가 맞는지 확인한다. 검증: 새 테스트 통과

## 2. 다변량·외생 변수 검증

- [x] 2.1 `recommend_model`에 `feature_columns`, `exog_columns` 인자를 추가한다. 다변량 리샘플, 80/20 분할, 후보별 열 배치, `get_predict(exog_data=...)`를 구현한다. 검증: 단변량 기존 테스트 4개 통과(회귀 없음)
- [x] 2.2 `recommended_params`와 `estimate_train_seconds`에 특징 변수와 변수 수를 넘긴다. 외생 변수는 파라미터 추천에 넘기지 않는다.

  검증: 스펙 "하이퍼파라미터 추천은 분석 유형에 따라 다르게 계산한다"의 시나리오 테스트
  - 트리 후보는 특징 변수 반영 여부에 따라 추천과 근거가 달라진다.
  - ETS는 특징 변수가 있어도 단변량과 같은 값을 추천한다.
  - 특징 변수가 있으면 예상 시간이 늘어난다.
- [x] 2.3 특징 변수가 있을 때 VectorAR을 기본 후보에 추가한다. 검증: 스펙의 "특징 변수 지정/없음" 시나리오 테스트
- [x] 2.4 `ForecastModel.recommend_model` 시그니처를 계약대로 확장한다: `(train_df, target_column, horizon, feature_columns=None, exog_columns=None, algorithms=None)`. 검증: 키워드 호출 테스트
- [x] 2.5 스펙 시나리오 테스트를 추가한다. 특징 변수 지연 의존 데이터에서 트리 후보의 오차가 줄어드는지, 외생 변수에서 Arima가 exog를 받는지, ETS가 exog 열과 함께 정상 동작하는지 확인한다. 검증: `python -m pytest tests/dashboard/test_model_select.py -q` 통과

## 3. 벤치마크와 판단

- [x] 3.1 scratchpad에 다변량 벤치마크 스크립트를 작성한다. 데이터는 SKAB 8센서, `data/multivariate` 3종, Walmart exog이고, 비교 대상은 (A) 단변량 검증, (B) 다변량 검증, (B') VectorAR 제외다. 검증: 소수 시계열 smoke run 성공
- [x] 3.2 전체 벤치마크를 실행하고 데이터셋별 중앙값·평균 MASE, 승패, Wilcoxon p를 정리한다. 검증: 결과 표 산출
- [x] 3.3 VectorAR 추가 여부를 결정해 코드와 스펙에 반영한다. 손해이면 2.3을 되돌리고 스펙의 해당 요구사항을 제거한다. 검증: 스펙과 코드가 결정과 일치
- [x] 3.4 `model_select.py` docstring에 다변량 벤치마크 결과를 요약한다. 검증: `python -m pytest tests/dashboard -q` 전체 통과
