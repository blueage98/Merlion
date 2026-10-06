# Design

## Context

- 현재 추천은 `ForecastModel.recommend_lgbm_params()`(`merlion/dashboard/models/forecast.py`) 하나뿐입니다. `TemporalResample`로 모델과 같은 방식으로 리샘플링한 뒤 ACF를 계산하고, `SeasonalityLayer.detect_seasonality()`로 유의한 주기를 찾습니다. 결과는 추천값과 근거 통계를 담은 dict입니다.
- 팝업(`pages/forecast.py`의 `create_param_confirm_modal`)은 `maxlags`/`max_forecast_steps` 두 입력란이 고정 ID로 박혀 있고, `callbacks/forecast.py`의 `confirm_train_settings`가 `tuned_algorithms`에 있는 알고리즘에만 팝업을 엽니다. Confirm하면 `trigger["overrides"]`로 값을 학습 콜백에 넘기고 설정 표에도 반영합니다.
- 설정 표의 값은 문자열이고 `ModelMixin.parse_parameters()`가 타입별로 변환합니다. 튜플(`order`)은 JSON 리스트로, Enum/문자열 선택지는 이름으로 파싱됩니다.
- 대시보드에서 보이는 알고리즘별 파라미터(기본값):
  - Arima: `order=(4,1,2)`, `seasonal_order=(0,0,0,0)`
  - Sarima: `order=(4,1,2)`, `seasonal_order=(2,0,1,24)`
  - ETS: `error=add`, `trend=add`, `damped_trend=True`, `seasonal=add`, `seasonal_periods=None`
  - Prophet: `yearly/weekly/daily_seasonality=auto`, `seasonality_mode=additive`
  - VectorAR: `maxlags=None`(학습 시 `max(min(20, n//10), max_forecast_steps)`)
  - RandomForest/ExtraTrees: LGBM과 같은 `maxlags`, `max_forecast_steps`, `prediction_stride`
  - DefaultForecaster: `max_forecast_steps`, `granularity`(내부적으로 단변량은 AutoETS, 다변량·exog는 LGBMForecaster)
- 추천 계산은 Train 버튼의 동기 콜백 안에서 실행되므로(학습 콜백만 background) 수 초 안에 끝나야 합니다.
- 리샘플링은 Python 3.14에서 깨져 있어 기존 테스트가 `xfail` 마커를 씁니다. 새 테스트도 같은 마커를 씁니다.

## Goals / Non-Goals

**Goals:**
- 알고리즘마다 그 모델 구조에 맞는 추정 원리로 핵심 하이퍼파라미터를 추천하고, 근거(검정 통계, 정보 기준, 강도 값)를 사람이 읽을 수 있는 문장으로 보여 줍니다.
- 알고리즘을 추가할 때 추천 함수 하나를 등록하는 것만으로 팝업에 나타나게 합니다.
- 공통 통계(리샘플링, ACF, 주기 검출, STL 강도, 예측 가능 구간)는 한 곳에서 계산해 재사용합니다.

**Non-Goals:**
- 교차 검증이나 백테스트 기반의 전역 하이퍼파라미터 최적화(예: `learning_rate`, `n_estimators`, `max_depth`, `min_samples_split`, Prophet `changepoint_prior_scale`). 트리 모델의 용량 파라미터는 데이터 통계만으로는 근거 있게 추천하기 어렵고, 학습 수 회가 필요해 팝업 응답 시간을 넘습니다.
- AutoETS, AutoProphet 추천. 이들은 내부에서 주기와 구성 요소를 정보 기준으로 이미 탐색합니다.
- exog 변수(`exog_aggregation_policy` 등)에 대한 추천.
- Prophet `stan_backend` 학습 오류 수정(보류된 별도 문제).

## Decisions

### 1. 추정 원리: "모델 구조가 가정하는 데이터 특성"을 측정해 파라미터에 대응시킨다

각 파라미터가 모델 안에서 무엇을 표현하는지에 따라 추정 방법을 고릅니다. 크게 세 가지 원리를 씁니다.

- **(A) 자기상관 구조 측정** — 파라미터가 "얼마나 먼 과거를 보는가 / 얼마나 먼 미래까지 믿을 수 있는가"를 뜻할 때. ACF 감쇠와 계절 주기 검출을 씁니다.
- **(B) 통계적 가설 검정·분해 강도** — 파라미터가 "구조적 특성의 유무"(단위근, 추세, 계절성, 곱셈성)를 뜻할 때. KPSS 검정, STL 분해 강도, 진폭–수준 상관을 씁니다.
- **(C) 정보 기준(AICc/BIC) 모델 선택** — 파라미터가 "모형의 차수"이고, 후보 모형을 직접 적합해 비교할 수 있을 때. 과적합을 벌점으로 다루는 정보 기준을 씁니다. 비용이 크므로 탐색 범위와 데이터 길이를 제한합니다.

알고리즘별 적용:

| 알고리즘 | 파라미터 | 원리 | 방법 |
|---|---|---|---|
| 공통 | `max_forecast_steps` | A | ACF가 임계값(0.5) 아래로 처음 떨어지기 직전 시차(기존 LGBM 규칙) |
| LGBM / RF / ExtraTrees | `maxlags` | A | 예측 구간보다 긴 유의 주기 중 ACF가 가장 높은 주기(기존 LGBM 규칙) |
| Arima | `d` | B | KPSS(수준 정상성) 검정을 반복 적용, 기각되는 동안 차분(최대 2) |
| Arima | `p`, `q` | C | 차분 후 시계열에 대해 Hyndman–Khandakar 단계적(stepwise) 탐색, AICc 최소 |
| Sarima | `m` | A | 유의한 지배 주기 |
| Sarima | `D` | B | STL 계절 강도 F_s = max(0, 1 − Var(R)/Var(S+R)) > 0.64 이면 1 |
| Sarima | `P`, `Q` | C | 비계절 (p, d, q)를 먼저 정한 뒤, (P, Q) ∈ {0,1}² 네 후보를 최근 20주기 데이터로 학습해 AICc 최소 |
| ETS | `seasonal_periods`, `seasonal` 유무 | A | 유의한 지배 주기 |
| ETS | `seasonal` add/mul | B | 주기별 수준(평균)과 계절 진폭(표준편차)의 Spearman 상관 ≥ 0.5, p < 0.05, 데이터가 모두 양수이면 `mul` |
| ETS | `trend`, `damped_trend` | B | STL 추세 강도 F_t = max(0, 1 − Var(R)/Var(T+R)) ≥ 0.5 이면 `add` + `damped=True` |
| ETS | `error` | — | 항상 `add`(곱셈 오차는 양수 데이터·수준 비례 분산이 필요하고 수치적으로 불안정. 근거에 명시) |
| Prophet | `yearly/weekly/daily_seasonality` | A+B | 관측 가능성(데이터 길이 ≥ 2주기, 간격 < 주기/2) + 해당 달력 주기 근처(±10%)에서 ACF가 유의 |
| Prophet | `seasonality_mode` | B | ETS와 같은 진폭–수준 기준 |
| VectorAR | `maxlags` | C | statsmodels `VAR.select_order`로 AIC/BIC/HQIC 계산, BIC 최소 선택. 단변량이면 `ar_select_order`(BIC) |
| DefaultForecaster | `granularity` | — | `TemporalResample`이 추론한 간격 |

선택 이유와 대안:

- **Arima d에 KPSS를 쓰는 이유**: ADF는 귀무가설이 "단위근 있음"이라 검정력이 약할 때 과다 차분을 피하기 어렵습니다. auto.arima(`ndiffs`)와 같이 KPSS를 기본으로 하고, ADF p값은 근거에 참고로만 표시합니다. 대안인 "ACF가 천천히 감쇠하면 차분" 같은 육안 규칙은 기준이 모호해 제외했습니다.
- **p, q에 정보 기준을 쓰는 이유**: ACF/PACF 절단점으로 p, q를 읽는 Box–Jenkins 방식은 순수 AR/MA에서만 명확하고, ARMA 혼합형에서는 판단이 모호합니다. AICc는 적합도와 모수 개수를 함께 보므로 자동화에 적합합니다. 전체 격자(6×6=36회 적합) 대신 stepwise(보통 10~20회)로 적합 횟수를 줄입니다. 대안인 `pmdarima.auto_arima`는 새 의존성이 생겨 쓰지 않습니다.
- **Sarima D에 계절 강도를 쓰는 이유**: OCSB/CH 검정은 statsmodels에 없고, 계절 강도 0.64 기준은 auto.arima `nsdiffs`의 기본 방식과 같습니다.
- **ETS를 AIC 탐색이 아니라 강도 측정으로 정하는 이유**: AIC 탐색은 AutoETS가 이미 합니다. ETS 추천의 목적은 사용자가 직접 구성 요소를 고를 때 "왜 그런지"를 설명할 수 있는 값을 주는 것이므로, 각 구성 요소를 데이터 특성 하나에 대응시키는 쪽을 택했습니다. 추세가 있으면 감쇠 추세를 추천하는 것은 M 대회들에서 장기 예측 정확도가 더 좋았던 경험적 근거를 따른 것입니다.
- **VectorAR에 BIC를 쓰는 이유**: VAR은 차수당 모수가 k²개로 빠르게 늘어나 AIC가 과대 차수를 고르기 쉽습니다. 예측 목적이면 BIC의 간결한 차수가 보통 더 안정적입니다. 세 기준의 선택을 모두 근거에 보여 사용자가 바꿀 수 있게 합니다.
- **트리 모델은 `maxlags`/`max_forecast_steps`만 추천하는 이유**: Non-Goals 참고.

### 1-1. 지배 주기 선택: 반 주기 골짜기 조건

Sarima·ETS·Prophet이 쓰는 지배 주기는 "유의한 주기 중 ACF가 가장 높은 주기"이면서, ACF가 반 주기 지점보다 높아야 합니다(ACF[m] > ACF[m/2]). 천천히 변하는 데이터에서는 짧은 시차의 ACF가 원래 높아서, 이 조건이 없으면 288단계 주기 데이터에서 17이, AR(2)에서 2가 뽑혔습니다. 진짜 계절 주기는 반 주기 앞에 골짜기가 있습니다. 기존 LGBM 규칙은 이미 예측 구간 안의 주기를 제외하므로 결과가 바뀌지 않도록 이 조건을 적용하지 않습니다. 또 Merlion의 주기 검출이 백색 잡음에서 돌려주는 주기 1은 계절성이 아니므로 제외합니다.

### 2. 공통 예측 가능 구간(`max_forecast_steps`)

모든 추천 대상 알고리즘이 같은 ACF 감쇠 규칙을 씁니다. 알고리즘마다 다른 규칙(예: ARIMA 예측 분산이 무조건 분산의 일정 비율에 이르는 시점)을 쓰는 대안도 있지만, d ≥ 1이면 분산이 발산해 정의가 깨지고, 알고리즘을 바꿔 가며 비교할 때 테스트 구간이 달라지면 지표 비교가 어려워집니다. 대시보드는 `max_forecast_steps`로 테스트 구간을 자르므로(`ForecastModel.train`), 같은 데이터에서는 같은 구간으로 비교되도록 통일합니다.

### 2-1. 예외: 차분하는 ARIMA·SARIMA의 예측 구간

구현 후 `example.csv`(1분)의 kpi를 Arima로 학습했더니 예측이 600 근처에서 평평했습니다. 공통 규칙은 `max_forecast_steps=258`을 추천했지만, 테스트 RMSE(163.9)는 정답 평균 같은 어떤 상수 예측(163.6)보다도 낫지 않았습니다. 하루 전 같은 시각 값만 써도 130.4였습니다.

- **원인:** 원래 값(수준)의 ACF가 258단계까지 0.5 이상인 것은 수준이 천천히 변하기 때문입니다. d = 1인 ARIMA가 실제로 예측하는 것은 차분한 값이고, 그 ACF는 lag 1~3에서만 유의했습니다(-0.11, -0.18, -0.10). 그 뒤로 예측한 차분은 0이 되어 예측이 현재 수준에서 평평해집니다.
- **결정:** Arima·Sarima가 차분(d ≥ 1 또는 D = 1)을 추천하면, 차분한 값의 |ACF|가 95% 유의 한계(1.96/√n, 차수 탐색에 쓴 최근 데이터 기준) 안으로 처음 들어가기 직전 시차를 `max_forecast_steps`로 씁니다. 계절 차분이 있으면 예측이 마지막 주기를 반복하므로 최소 한 주기(m)를 보장합니다. 차분하지 않으면(d = D = 0) 공통 규칙을 그대로 씁니다.
- **결과:** `example.csv`는 258에서 3으로, `machine_temperature_system_failure.csv`는 110에서 7로 바뀝니다. 이 구간을 넘으면 ARIMA 예측은 현재 수준을 반복할 뿐이라는 뜻입니다.
- **안내 노트:** 같은 사건에서 실제로 예측에 쓸 수 있던 정보는 1일 주기(1440)였습니다. Arima, 또는 주기가 상한보다 길어 계절 항을 뺀 Sarima는 그 주기를 표현하지 못합니다. 그래서 팝업 노트로 주기와 대안을 안내합니다. 주기가 24 이하이면 Sarima를, 그보다 길면 리샘플링이나 LGBMForecaster/RandomForestForecaster·Prophet을 권합니다.
- **2절과의 관계:** 2절의 "같은 데이터에서는 같은 테스트 구간" 원칙에서 벗어납니다. 하지만 차분하는 ARIMA에 원래 값 기준 구간을 쓰면, 사용자가 "그 구간까지 예측할 수 있다"고 오해하게 됩니다. 그래서 정직한 구간을 우선했습니다.

### 3. 모듈 구조와 등록 테이블

- 새 모듈 `merlion/dashboard/models/recommend.py`에 다음을 둡니다.
  - `ParamRecommendation`(dataclass): `name`, `value`, `kind`(`"int"`, `"choice"`, `"int_tuple"`, `"str"`), `basis`(문장), `choices`(선택지), `tuple_len`, `optional`(빈 값 허용, 예: ETS `seasonal_periods=None`), `is_steps`(기간 환산 여부). DefaultForecaster `granularity`(예: `"5min"`)를 위해 `"str"` 종류를 추가했습니다.
  - `Recommendation`(dataclass): `params: List[ParamRecommendation]`, `n_points`, `granularity`, `notes`.
  - `SeriesStats`: 리샘플링한 대상 변수, ACF, 유의 주기 목록, STL 강도 등을 필요할 때 한 번만 계산해 캐시하는 도우미. 추천 함수들이 같은 통계를 다시 계산하지 않게 합니다.
  - 알고리즘별 추천 함수 `recommend_<algo>(stats, ...) -> Recommendation`와 등록 테이블 `RECOMMENDERS = {"Arima": recommend_arima, ...}`.
- `ForecastModel.tuned_algorithms`는 `list(RECOMMENDERS)`로 대체하고, 공통 진입점 `ForecastModel.recommend_params(algorithm, train_df, target_column, feature_columns)`를 둡니다.
- `recommend_lgbm_params()`는 반환 형식을 유지하고(기존 테스트), 트리 모델 추천 함수가 그 결과를 `Recommendation`으로 바꿔 씁니다. RF/ExtraTrees는 같은 함수를 등록만 합니다.
- 대안: `forecast.py`에 메서드로 계속 추가. 파일이 커지고 통계 캐시를 공유하기 어려워 분리했습니다.

### 4. 팝업 일반화

- 고정 입력란 두 개를 없애고, 콜백이 채우는 컨테이너 `forecasting-confirm-inputs`를 둡니다. 입력란 ID는 Dash 패턴 매칭 `{"type": "forecasting-confirm-param", "name": <파라미터>}`로 만들고, Confirm 콜백은 `State({"type": ..., "name": ALL}, "value")`와 `"id"`로 읽습니다.
- `kind`별 컴포넌트: `int` → `dcc.Input(type="number", min=1, step=1)`, `choice` → `dcc.Dropdown`(`"None"` 포함 가능), `int_tuple` → `dcc.Input(type="text")`.
- 입력 검증은 순수 함수 `parse_confirmed_values(specs, values) -> (overrides, errors)`로 분리해 단위 테스트합니다. 튜플은 `(1, 0, 1)`/`[1,0,1]` 형식을 모두 받고 길이와 음이 아닌 정수를 확인합니다. `"None"`은 Python `None`, `"True"/"False"`는 bool로 바꿉니다.
- 확인한 값은 지금처럼 `trigger["overrides"]`로 넘기고 설정 표에 문자열로 반영합니다. 튜플은 JSON 직렬화를 위해 리스트로 넘깁니다(statsmodels가 리스트를 받고, `parse_parameters`도 튜플을 리스트로 파싱하므로 동작이 같습니다).
- 팝업 확인 대상 파라미터 목록은 추천 결과에 담겨 오므로, 팝업을 열 때 함께 `dcc.Store`에 저장해 Confirm 시 검증 규격으로 씁니다.

### 5. 시간 제한

- 정보 기준 탐색(Arima, Sarima, VectorAR)은 리샘플링한 데이터의 최근 `ic_max_points = 2000`점만 씁니다.
- Arima stepwise 탐색은 최대 `ic_max_fits = 15`회 적합으로 제한합니다. 처음에는 30회로 잡았지만, `example.csv`(1분, 7만 점)에서 차수가 (5,1,5)까지 올라가며 11.5초가 걸려 15회로 낮췄습니다(4.4초, (3,1,4) 선택).
- Sarima는 `sarima_max_period = 24`보다 긴 주기의 계절 항을 생략하고, 근거에 "더 큰 간격으로 리샘플링하거나 다른 알고리즘을 쓰라"는 안내를 넣습니다.
  - 처음에는 상한을 100으로 잡았지만, 구현 중 측정해 보니 (P, Q) 네 후보의 AICc 탐색 시간이 m에 따라 빠르게 늘었습니다: m=24 약 4초, 36은 8초, 48은 15초, 100은 1분 이상(`simple_differencing=True`, 최근 20주기 기준). 그래서 AICc 탐색을 유지하고 상한을 24로 낮췄습니다(시간 단위 데이터의 일 주기, 월 단위 데이터의 연 주기 12 포함).
  - 대안으로 검토한 것: (P, Q)를 차분한 시계열의 시차 m에서 PACF/ACF 유의성으로 정하는 Box–Jenkins 방식(상한 100 유지 가능하지만 정보 기준 비교가 아님), 그리고 m ≤ 24는 AICc, 그 이상은 ACF/PACF로 나누는 혼합 방식. 사용자 결정으로 AICc + 상한 24를 택했습니다.
  - 계절 후보 학습에는 `simple_differencing=True`(차분을 먼저 하고 상태에서 계절 시차를 빼서 약 4배 빠름, 후보 순위는 같음)와 `sarima_cycles = 20`(최근 20주기)을 씁니다.
- 상수는 `ForecastModel`의 클래스 변수로 두어 조정할 수 있게 합니다(`lgbm_acf_threshold`와 같은 방식).
- STL 분해는 주기와 데이터 길이에 비례해 느려집니다(주기 1440, 1만 4천 점에 10초). 주기가 `SeriesStats.stl_max_period = 100`보다 길면 연속한 점들을 평균 내어 주기를 100단계 이하로 줄인 뒤 분해합니다. 추세·계절 강도와 진폭–수준 상관은 분산 비율이라 해상도를 낮춰도 거의 변하지 않습니다.
- 실측(학습 데이터 80%): `machine_temperature_system_failure.csv`(5분, 1만 8천 점)는 9개 알고리즘 모두 2초 이내, `example.csv`(1분, 7만 점)는 Sarima 6.0초, Arima 4.4초, 나머지 2초 이내입니다.

## Risks / Trade-offs

- [STL은 `period ≥ 2`이고 데이터가 2주기 이상이어야 함] → 주기가 없거나 짧은 데이터는 강도를 0으로 보고 해당 구성 요소를 "없음"으로 추천하며 근거에 이유를 적습니다.
- [Prophet 연 주기는 시간 단위 데이터에서 시차가 8760으로 ACF 계산이 큼] → 달력 주기를 확인할 때 데이터를 그 주기의 1/24 정도 간격으로 평균 집계한 뒤 ACF를 계산합니다(예: 연 주기는 일 단위로 집계).
- [정보 기준 탐색 중 SARIMAX 수렴 경고·실패] → 경고는 억제하고, 적합에 실패한 후보는 건너뜁니다. 모든 후보가 실패하면 그 파라미터만 추천에서 뺍니다(spec: 부분 실패).
- [ETS 곱셈 계절성·추세 강도 임계값은 경험값] → 임계값과 측정값을 근거에 함께 보여 사용자가 판단할 수 있게 하고, 클래스 변수로 조정 가능하게 둡니다.
- [불규칙 간격 데이터(`example.csv`)에서는 리샘플링 보간이 ACF를 부풀릴 수 있음] → 모델도 같은 리샘플링을 쓰므로 추천과 학습의 입력은 일치합니다. 팝업에 리샘플링 후 점 수를 표시해 사용자가 알 수 있게 합니다(기존 동작과 같음).
- [VAR 차수 선택 실패] 상수이거나 공선성이 있는 feature(예: `example.csv`의 `kpi_label`)는 공분산 행렬을 특이하게 만듭니다 → `maxlags`만 추천에서 빼고 팝업 노트에 이유를 적습니다(spec: 부분 실패).
- [ETS의 긴 계절 주기] 분 단위 데이터에서는 `seasonal_periods`가 1440~1969처럼 길게 추천됩니다. statsmodels ETS는 계절 초기값을 주기 수만큼 모수로 추정해서, 학습 시간이 주기가 두 배가 될 때마다 약 6배 늘어납니다(2000점 기준 m=24 0.5초, 48 2.6초, 96 12.8초, 192 84초). `machine_temperature_system_failure.csv`에서 `seasonal_periods=1969`로 학습해 보니 20분이 지나도 끝나지 않았습니다 → 사용자 결정으로 주기는 그대로 추천하되, `ets_slow_period = 48`보다 길면 근거와 팝업 노트에 "학습이 매우 오래 걸릴 수 있으니 더 큰 간격으로 리샘플링하거나 주기를 줄이거나 `seasonal`을 `None`으로 바꾸라"는 경고를 보여 줍니다. Sarima처럼 상한을 두어 계절 항을 빼는 방식도 검토했지만 택하지 않았습니다.
- [동기 콜백 지연] → 5절의 제한으로 실험 데이터(2만 점대)에서 수 초 이내를 목표로 하고, 구현 중 `machine_temperature_system_failure.csv`로 시간을 측정해 확인합니다.

## Migration Plan

대시보드 내부 기능이고 저장 형식 변경이 없어 별도 마이그레이션은 없습니다. 문제가 생기면 등록 테이블에서 해당 알고리즘을 빼면 그 알고리즘은 팝업 없이 예전처럼 학습됩니다.
