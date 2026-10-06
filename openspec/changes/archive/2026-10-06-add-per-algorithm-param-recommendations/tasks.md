# Tasks

## 1. 공통 기반과 등록 테이블

- [x] 1.1 `merlion/dashboard/models/recommend.py`에 `ParamRecommendation`, `Recommendation` dataclass와 `SeriesStats`(리샘플링, ACF, 유의 주기, STL 추세·계절 강도, 진폭–수준 상관을 지연 계산·캐시)를 추가하고, `tests/dashboard/test_param_recommendation.py`에 합성 데이터로 STL 강도·주기 검출 단위 테스트를 추가해 통과를 확인한다
- [x] 1.2 공통 `max_forecast_steps` 규칙(ACF 감쇠)을 `SeriesStats`의 함수로 옮기고, `recommend_lgbm_params()`가 이를 쓰도록 바꾼 뒤 `pytest tests/dashboard/test_lgbm_recommendation.py`가 그대로 통과하는지 확인한다
- [x] 1.3 트리 모델 추천 함수(`recommend_lgbm_params()` 결과를 `Recommendation`으로 변환)와 등록 테이블 `RECOMMENDERS`, `ForecastModel.recommend_params()` 진입점을 추가하고, `tuned_algorithms`를 등록 테이블에서 만들도록 바꾼 뒤 LGBMForecaster 추천 결과가 바뀌지 않는다는 테스트를 추가해 통과를 확인한다
- [x] 1.4 시간 제한 상수(`ic_max_points`, `ic_max_fits`, `sarima_max_period`, ETS·Prophet 임계값)를 `ForecastModel` 클래스 변수로 추가하고 docstring에 의미를 적는다

## 2. 확인 팝업 일반화

- [x] 2.1 입력 검증 순수 함수 `parse_confirmed_values(specs, values)`를 추가하고, 정수(0·음수 거부), 선택지(`None`/`True`/`False` 변환), 튜플(길이·형식·음수 거부) 경우를 단위 테스트로 확인한다
- [x] 2.2 `pages/forecast.py`의 고정 입력란을 `forecasting-confirm-inputs` 컨테이너와 추천 규격 저장용 `dcc.Store`로 바꾸고, `kind`별 입력 컴포넌트를 만드는 함수를 추가한다. 레이아웃이 오류 없이 만들어지는지 `tests/dashboard`의 레이아웃 생성 테스트로 확인한다
- [x] 2.3 `_create_recommendation_content()`를 `Recommendation`의 파라미터 목록으로 표(파라미터, 현재값, 추천값, 기간, 근거)를 그리도록 바꾸고, 단계 수 파라미터에만 기간이 표시되는지 테스트한다
- [x] 2.4 `confirm_train_settings`를 패턴 매칭 State로 바꿔 `recommend_params()` 호출 → 팝업 열기, Confirm 시 검증·overrides·설정 표 반영, Cancel, 추천 실패 시 바로 학습 흐름을 구현하고, 콜백 함수를 직접 호출하는 테스트로 네 경로를 확인한다
- [x] 2.5 대시보드를 띄워 LGBMForecaster로 Train → 팝업 → Confirm 흐름이 이전과 같이 동작하는지 브라우저에서 확인한다

## 3. RandomForest / ExtraTrees

- [x] 3.1 `RandomForestForecaster`, `ExtraTreesForecaster`를 트리 모델 추천 함수로 등록하고, 주기 24 데이터에서 `maxlags=24`가 추천되는 테스트를 추가해 통과를 확인한다

## 4. Arima

- [x] 4.1 KPSS 반복 검정으로 d(최대 2)를 정하는 함수를 추가하고, 랜덤 워크에서 d=1, 정상 AR(2)에서 d=0, 이중 누적합에서 d=2인지 테스트한다
- [x] 4.2 차분 차수를 고정한 Hyndman–Khandakar stepwise AICc 탐색(p, q ∈ 0..5, 최근 `ic_max_points`점, 최대 `ic_max_fits`회 적합, 실패 후보 건너뜀)을 추가하고, AR(2) 데이터에서 p ≥ 1, 적합 횟수가 상한 이하인지 테스트한다
- [x] 4.3 `recommend_arima`(order + 공통 max_forecast_steps, 근거에 KPSS·ADF p값과 AICc 포함)를 등록하고, 2만 점 데이터에서 추천이 수 초 안에 끝나며 사용한 점 수가 결과에 담기는지 테스트한다

## 5. Sarima

- [x] 5.1 계절 주기 m 선택, STL 계절 강도로 D 결정, 계절 차분 후 d 결정, 비계절 차수 AICc 탐색 후 (P, Q) ∈ {0,1}²를 최근 `sarima_cycles` 주기로 AICc 탐색을 구현한 `recommend_sarima`를 등록하고, 1시간 간격·주기 24 데이터에서 `m=24, D=1`인지 테스트한다
- [x] 5.2 주기가 없거나 `sarima_max_period`(24)보다 긴 경우 `seasonal_order=(0,0,0,0)`과 이유가 담긴 근거를 반환하는지, 주기 288 데이터로 테스트한다

## 6. ETS

- [x] 6.1 `recommend_ets`(seasonal_periods, seasonal add/mul/None, trend add/None, damped_trend, error=add + 공통 max_forecast_steps)를 등록하고, 추세+곱셈 계절성 데이터와 백색 잡음 데이터에서 spec의 추천값이 나오는지 테스트한다
- [x] 6.2 추천값 그대로 ETS 학습이 성공하는지(`ForecastModel().train`) 주기 12 데이터로 테스트한다
- [x] 6.3 `ets_slow_period`(48)보다 긴 추천 주기에 학습 시간 경고(근거·팝업 노트)를 붙이고, 주기 288 데이터에서 경고가 나오고 주기 12 데이터에서는 나오지 않는지 테스트한다

## 7. Prophet

- [x] 7.1 달력 주기(일·주·연)별 관측 가능성 판정과 집계 후 ACF 유의성 검사를 구현한 `recommend_prophet`(세 계절성 bool + seasonality_mode + 공통 max_forecast_steps)를 등록하고, 1시간 간격 8주 일 주기 데이터에서 `daily=True, yearly=False`인지 테스트한다(Prophet 학습 오류와 무관하게 추천만 검증)

## 8. VectorAR

- [x] 8.1 `feature_columns`를 받아 다변량이면 `VAR.select_order`(BIC 선택, 근거에 AIC/BIC/HQIC 차수), 단변량이면 `ar_select_order`로 `maxlags`를 추천하는 `recommend_vector_ar`를 등록하고, 차수 2인 2변량 VAR 데이터에서 `maxlags=2`, 단변량 AR(3)에서 3 근처인지 테스트한다

## 9. DefaultForecaster

- [x] 9.1 `recommend_default`(공통 max_forecast_steps + 추론한 granularity)를 등록하고, 5분 간격 데이터에서 granularity가 5분으로 추천되고 그 값으로 학습이 성공하는지 테스트한다

## 10. 통합 확인

- [x] 10.1 `python -m pytest tests/dashboard -q` 전체를 실행해 기존·신규 테스트가 모두 통과(Python 3.14 xfail 제외)하는지 확인한다
- [x] 10.2 `machine_temperature_system_failure.csv`(5분)와 `example.csv`(불규칙 1분)로 등록된 9개 알고리즘 각각의 추천 시간을 측정해 모두 수 초 이내인지 확인하고, 넘는 경우 시간 제한 상수를 조정한다
- [x] 10.3 대시보드에서 Sarima, ETS, VectorAR로 Train → 팝업(표·입력란 종류) → 값 수정 → Confirm → 설정 표 반영·학습 완료 흐름과, AutoETS가 팝업 없이 학습되는 것을 브라우저에서 확인한다

## 11. 차분하는 ARIMA·SARIMA의 예측 구간과 주기 안내 (example.csv에서 ARIMA 예측이 평평했던 문제)

- [x] 11.1 `recommend_arima_horizon`을 추가해, 차분(d ≥ 1 또는 D = 1)이 있으면 차분한 값의 |ACF|가 95% 유의 한계 안으로 처음 들어가기 직전 시차(D = 1이면 최소 m)를 `max_forecast_steps`로 추천하게 한다. 랜덤 워크에서 3 이하, 정상 AR(2)에서 공통 규칙과 같은 값, 주기 24 Sarima에서 24 이상인지 테스트로 확인한다
- [x] 11.2 Arima와 상한을 넘는 주기의 Sarima에 표현하지 못하는 계절 주기 안내 노트를 추가하고, 주기 24(Sarima 권장), 주기 288(리샘플링·LGBM·Prophet 권장), 주기 없음(노트 없음)을 테스트로 확인한다
- [x] 11.3 `example.csv`와 `machine_temperature_system_failure.csv`에서 Arima·Sarima 추천을 다시 계산해 `max_forecast_steps`가 3, 7로 바뀌고 1일 등의 주기 안내가 나오는지 확인한다

