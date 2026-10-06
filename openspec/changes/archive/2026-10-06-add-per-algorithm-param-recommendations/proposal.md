# Proposal

## Why

지금 대시보드의 Forecasting 탭은 LGBMForecaster에 대해서만 학습 전에 `maxlags`/`max_forecast_steps` 추천값을 근거와 함께 보여 줍니다. 나머지 알고리즘은 기본값(예: Arima `order=(4,1,2)`, Sarima `seasonal_order=(2,0,1,24)`)으로 바로 학습되기 때문에, 데이터의 주기·추세·정상성과 맞지 않는 설정으로 학습되기 쉽습니다. 알고리즘마다 핵심 하이퍼파라미터와 그 최적값을 추정하는 원리가 다르므로(자기상관 감쇠, 단위근 검정, 정보 기준, 계절 강도 등), 알고리즘별로 원리에 맞는 추천 함수가 필요합니다.

## What Changes

- 알고리즘별 추천 함수를 추가하고, `{알고리즘: 추천 함수}` 등록 테이블로 관리합니다. 추천 대상 알고리즘과 파라미터는 다음과 같습니다.
  - **LGBMForecaster / RandomForestForecaster / ExtraTreesForecaster** (자기회귀 트리 모델): `maxlags`, `max_forecast_steps` — 기존 ACF 기반 규칙을 공유합니다.
  - **Arima**: `order (p, d, q)` — 단위근 검정으로 d, 정보 기준(AICc) 탐색으로 p·q를 정합니다.
  - **Sarima**: `order`, `seasonal_order (P, D, Q, m)` — 계절 주기 m 검출, 계절 강도로 D, 정보 기준으로 P·Q를 정합니다.
  - **ETS**: `error`, `trend`, `damped_trend`, `seasonal`, `seasonal_periods` — STL 분해의 추세/계절 강도와 진폭–수준 관계로 정합니다.
  - **Prophet**: `yearly/weekly/daily_seasonality`, `seasonality_mode` — 데이터 길이·샘플링 간격·달력 주기의 유의성으로 정합니다.
  - **VectorAR**: `maxlags` — VAR 차수 선택 정보 기준(BIC)으로 정합니다.
  - **DefaultForecaster**: `max_forecast_steps`, `granularity`.
  - 모든 추천 대상 알고리즘에 `max_forecast_steps`를 공통 "예측 가능 구간" 규칙(ACF 감쇠)으로 추천합니다.
- 학습 전 확인 팝업을 일반화합니다. 고정된 `maxlags`/`max_forecast_steps` 입력란 대신, 알고리즘이 추천한 파라미터 목록에 맞춰 입력란(정수, 선택지, 튜플)을 동적으로 만들고, 파라미터마다 현재값·추천값·근거를 표로 보여 줍니다.
- 추천 계산은 팝업 응답 시간 안에 끝나도록 데이터 길이와 모델 적합 횟수를 제한합니다. 추천이 실패하면 지금처럼 팝업 없이 기존 설정으로 학습합니다.
- AutoETS, AutoProphet은 내부에서 이미 자동 탐색을 하므로 추천 대상에서 제외합니다(팝업 없이 바로 학습).

## Capabilities

### New Capabilities
- `forecast-param-recommendation`: Forecasting 탭에서 학습 전에 알고리즘별 하이퍼파라미터 추천값과 근거를 계산해 보여 주고, 사용자가 확인·수정한 값으로 학습하는 기능. 기존 LGBMForecaster 추천 동작도 이 capability로 편입합니다.

### Modified Capabilities
(없음 — 기존 spec `sample-data-selection`은 영향을 받지 않습니다.)

## Impact

- **코드**:
  - `merlion/dashboard/models/forecast.py`: `tuned_algorithms`를 등록 테이블 기반으로 바꾸고, 공통 진입점을 추가합니다.
  - 새 모듈 `merlion/dashboard/models/recommend.py`(가칭): 알고리즘별 추천 함수와 공통 통계 유틸리티를 둡니다.
  - `merlion/dashboard/callbacks/forecast.py`: `confirm_train_settings`, `_create_recommendation_content`를 파라미터 목록 기반으로 일반화합니다.
  - `merlion/dashboard/pages/forecast.py`: 확인 팝업의 입력란을 동적으로 바꿉니다.
- **테스트**: `tests/dashboard/`에 알고리즘별 추천 테스트를 추가하고, 기존 `test_lgbm_recommendation.py`는 계속 통과해야 합니다.
- **의존성**: 새 외부 의존성은 추가하지 않습니다(statsmodels, scipy, numpy만 사용).
- **호환성**: `ForecastModel.recommend_lgbm_params()`의 반환 형식은 유지합니다. 팝업에 보이는 파라미터가 알고리즘마다 달라지는 것 외에 학습 흐름(Train → 팝업 → Confirm/Cancel)은 같습니다.
