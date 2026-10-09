## Context

- `recommend_model`은 `train_df.loc[:, [target_column]]`만 리샘플링해서 후보를 검증한다([model_select.py](../../../merlion/dashboard/models/model_select.py)). `recommended_params`도 특징 변수 없이 호출한다.
- 대시보드 학습(`ForecastModel.train`, [models/forecast.py](../../../merlion/dashboard/models/forecast.py))은 열을 `[target] + features + exog` 순서로 배치하고 `target_seq_index`를 지정한다.
  - 모델이 외생 변수를 지원하면 exog를 떼어 내 `exog_data`로 넘기고, 이때 테스트 구간의 exog도 이어 붙인다.
  - 지원하지 않으면 exog 열을 학습 데이터에 그대로 남긴다.
- Merlion `ForecasterEnsemble._train_with_exog`는 다변량 학습 데이터와 `exog_data`를 그대로 `ForecastEvaluator.get_predict(train_vals, test_vals, exog_data=...)`에 넘긴다. `ModelSelector`는 대상 변수의 지표로 순위를 매긴다.
- 모델별 처리: ETS는 대상 변수만 쓰고, 외생 변수를 지원하지 않는다. Arima, Sarima, Prophet은 대상 변수와 외생 변수를 쓴다(`ForecasterExogBase`). 트리 모델과 LGBM은 모든 열의 지연값과 외생 변수를 쓴다. VectorAR은 모든 열을 쓴다.

## Goals / Non-Goals

**Goals:**
- 검증 조건을 대시보드 학습 조건과 일치시킨다. 열 구성, 대상 위치, 외생 변수 처리를 맞춘다.
- 단변량 입력의 결과는 그대로 유지한다(회귀 없음).
- 다변량 벤치마크로 개선 여부와 VectorAR 추가 여부를 수치로 판단한다.

**Non-Goals:**
- 여러 대상 변수를 동시에 예측하는 multivariate output은 다루지 않는다.
- 특징 변수 자체를 고르는 feature selection은 다루지 않는다.
- 검증 방식(80/20, MAE)은 바꾸지 않는다.
- 대시보드 UI는 바꾸지 않는다(`add-forecast-auto-train`에서 다룸).

## Decisions

1. **열 배치를 학습과 공유한다.**
   - `ForecastModel.train`의 열 정규화와 배치 로직을 함수로 분리한다. 이 로직은 열 이름의 int 변환, `[target] + features + exog` 순서, `target_seq_index`, 외생 변수 지원 여부에 따른 분리다.
   - `recommend_model`과 `train`이 이 함수를 함께 쓴다.
   - 대안은 `model_select.py`에 같은 로직을 다시 쓰는 것인데, 둘이 어긋날 위험이 바로 이번 문제의 원인이므로 택하지 않는다.
2. **외생 변수는 `ForecastEvaluator.get_predict(..., exog_data=exog_ts)`로 넘긴다.**
   - Merlion 앙상블과 같은 경로다. exog 시계열은 학습 데이터 전체(80%와 20% 모두)를 덮으므로 검증 구간 예측에 미리 알려진 값이 들어간다.
   - 외생 변수를 지원하지 않는 모델은 exog 열을 학습 데이터에 남긴다. `ForecastModel.train`과 같은 처리다.
3. **리샘플링은 다변량 전체에 한 번 적용한다.**
   - `TemporalResample`을 [target + features (+ exog)] TimeSeries에 적용하고, 그 결과를 80/20으로 나눈다. 모델 내부 기본 transform과 같은 동작이다.
   - 외생 변수는 별도 TimeSeries로 만든다. 모델이 학습 granularity에 맞춰 exog를 리샘플한다(Merlion 기본 동작).
4. **추천 파라미터와 시간 추정에 특징 변수를 넘긴다.**
   - `recommended_params(..., feature_columns)`를 호출한다.
   - `estimate_train_seconds(..., n_variables=학습에 들어가는 열 수)`로 계산한다. `ForecastModel.estimate_train_time`과 같은 기준이다.
5. **VectorAR은 조건부 기본 후보로 둔다.**
   - 특징 변수가 있을 때만 `DEFAULT_CANDIDATES`에 추가한다.
   - 벤치마크에서 VectorAR이 들어간 선택이 빠진 선택보다 중앙값 MASE가 나쁘고 유의하게 손해이면 추가하지 않는다. 이 경우 스펙의 해당 요구사항은 apply 단계에서 REMOVED로 갱신한다.
6. **시그니처는 키워드 인자로 확장한다.**
   - `recommend_model(train_df, target_column, horizon, cfg, feature_columns=None, exog_columns=None, algorithms=None, ...)` 형태로 넓힌다.
   - `ForecastModel.recommend_model(train_df, target_column, horizon, feature_columns=None, exog_columns=None, algorithms=None)`를 제공한다.
   - 반환 타입 `ModelRecommendation`의 필드는 바꾸지 않는다. 병렬 작업이 이 계약에 의존한다.

## 벤치마크 계획

- **데이터**
  - SKAB 무이상 데이터(`data/manufacturing/skab_anomaly_free.csv`, 1초 간격 8개 센서): 대상 센서별로 다른 센서를 특징 변수로 쓰고, rolling-origin 창을 만든다.
  - `data/multivariate/{energy_power, seattle_trail, solar_plant}`: 첫 열을 대상, 나머지를 특징 변수로 쓴다.
  - `data/walmart/walmart_mini.csv`: Store·Dept별 Weekly_Sales를 대상으로 하고, Temperature, Fuel_Price, CPI, Unemployment, IsHoliday를 외생 변수로 쓴다.
- **비교 대상**
  - (A) 현재 방식: 대상 변수만으로 검증해서 고른다.
  - (B) 이번 변경: 학습 조건과 같은 다변량·외생 검증으로 고른다.
  - (B') (B)에서 VectorAR을 뺀 버전.
- **평가 방법**: 모두 "고른 알고리즘을 대시보드와 같은 방식(특징·외생 변수 포함)으로 전체 학습 데이터에 학습"한 뒤, 테스트 구간 MASE로 평가한다.
- **판정 기준**: 중앙값과 평균 MASE, 시계열별 승패, Wilcoxon 검정을 본다.

## Risks / Trade-offs

- [특징 변수가 많으면 트리·VectorAR 후보의 검증 시간이 늘어남] → 학습 시간 추정에 변수 수를 반영해 기준 초과 후보는 제외한다.
- [외생 변수가 결측이면 예측이 실패할 수 있음] → 실패 후보로 처리하고 순위 뒤로 보낸다. 대시보드 학습에서도 같은 실패가 나므로 일관된다.
- [`ForecastModel.train` 리팩터링으로 기존 학습 경로가 회귀할 수 있음] → 기존 `tests/dashboard/test_forecast_model.py`를 모두 통과해야 한다.
- [다변량 벤치마크 데이터가 적어 결론이 약할 수 있음] → 결과를 데이터셋별로 보고한다. 유의하지 않으면 "학습 조건과 일치"라는 정합성 근거만으로 채택하고 그 사실을 명시한다.
