# forecast-param-recommendation Specification

## Purpose

대시보드 Forecasting 탭에서 학습을 시작하기 전에, 선택한 알고리즘의 핵심 하이퍼파라미터 추천값을 학습 데이터로부터 계산해 근거와 함께 보여 주고, 사용자가 확인하거나 수정한 값으로 학습하게 합니다.

## Requirements

### Requirement: 추천 대상 알고리즘은 학습 전에 확인 팝업을 연다
Train 버튼을 누르면, 선택한 알고리즘이 추천 대상(LGBMForecaster, RandomForestForecaster, ExtraTreesForecaster, Arima, Sarima, ETS, VectorAR, DefaultForecaster)인 경우 시스템은 SHALL 학습을 시작하지 않고 추천 설정 팝업을 연다. 추천 대상이 아닌 알고리즘(Prophet, AutoETS, AutoProphet 등)은 SHALL 팝업 없이 바로 학습을 시작한다.

추천 대상은 추천값이 최적화 없이 정한 기본값보다 나은 알고리즘으로 한정한다. Prophet은 2026-10-09 벤치마크(M4 Hourly 414개, 제조 데이터 85개 창)에서 추천값이 Prophet 자체의 `auto` 계절성보다 유의하게 나빴고(M4: 개선 183, 악화 231, Wilcoxon p = 0.0003), 후보 설정을 홀드아웃으로 고르도록 고쳐도 제조 데이터에서는 개선이 없어(machine temperature 9:9, SKAB 64개 모두 동일) 추천 대상에서 뺐다.

#### Scenario: 추천 대상 알고리즘 선택 후 Train
- **WHEN** 사용자가 알고리즘으로 Sarima를 선택하고 Train을 누른다
- **THEN** 추천 설정 팝업이 열리고, 사용자가 Confirm을 누르기 전에는 학습이 시작되지 않는다

#### Scenario: 자동 탐색 알고리즘 선택 후 Train
- **WHEN** 사용자가 알고리즘으로 AutoETS를 선택하고 Train을 누른다
- **THEN** 팝업 없이 알고리즘 설정 표의 값으로 바로 학습이 시작된다

### Requirement: 팝업은 파라미터별 현재값, 추천값, 근거를 보여 준다
팝업은 SHALL 해당 알고리즘이 추천한 파라미터마다 이름, 알고리즘 설정 표의 현재값, 추천값, 추천 근거를 한 행으로 보여 준다. 값이 시간 단계 수인 파라미터는 SHALL 학습 데이터의 샘플링 간격으로 환산한 기간도 함께 보여 준다. 팝업은 SHALL 추천 계산에 쓴 데이터 점 수와 샘플링 간격도 보여 준다.

#### Scenario: Arima 추천 표
- **WHEN** Arima에 대한 추천 팝업이 열린다
- **THEN** 표에 `order` 행이 있고, 그 행에는 현재값 `(4, 1, 2)`, 추천값, 그리고 차분 차수와 정보 기준 비교를 설명하는 근거가 표시된다

#### Scenario: 단계 수 파라미터의 기간 표시
- **WHEN** 1시간 간격 데이터로 `max_forecast_steps`가 24로 추천된다
- **THEN** 해당 행의 기간 열에 1일에 해당하는 기간이 표시된다

### Requirement: 사용자는 추천값을 수정해 확인할 수 있다
팝업은 SHALL 추천한 파라미터마다 추천값이 미리 채워진 입력란을 제공한다. 입력란은 SHALL 파라미터 종류에 맞춘다: 양의 정수는 숫자 입력, 정해진 선택지(예: `add`/`mul`/`None`, `True`/`False`)는 선택 목록, 정수 튜플(예: `order`)은 텍스트 입력. 형식에 맞지 않는 값으로 Confirm하면 시스템은 SHALL 학습을 시작하지 않고 팝업 안에 어떤 파라미터가 잘못되었는지 오류를 보여 준다.

#### Scenario: 잘못된 튜플 입력
- **WHEN** 사용자가 Sarima 팝업에서 `seasonal_order`를 `(1, 0, 1)`(원소 3개)로 바꾸고 Confirm을 누른다
- **THEN** 학습은 시작되지 않고, 팝업에 `seasonal_order`는 정수 4개여야 한다는 오류가 표시된다

#### Scenario: 정수 파라미터에 0 입력
- **WHEN** 사용자가 `maxlags`를 0으로 바꾸고 Confirm을 누른다
- **THEN** 학습은 시작되지 않고, 팝업에 `maxlags`는 양의 정수여야 한다는 오류가 표시된다

### Requirement: 확인한 값으로 학습하고 설정 표에 반영한다
Confirm을 누르면 시스템은 SHALL 팝업 입력란의 값을 알고리즘 설정 표의 값보다 우선해 학습에 쓰고, 그 값을 알고리즘 설정 표에도 반영한다. Cancel을 누르면 SHALL 학습을 시작하지 않고 설정 표도 바꾸지 않는다.

#### Scenario: Confirm
- **WHEN** ETS 팝업에서 사용자가 `seasonal_periods`를 24로 두고 Confirm을 누른다
- **THEN** ETS가 `seasonal_periods=24`로 학습되고, 알고리즘 설정 표의 `seasonal_periods` 값이 24로 바뀐다

#### Scenario: Cancel
- **WHEN** 사용자가 팝업에서 Cancel을 누른다
- **THEN** 학습이 시작되지 않고 알고리즘 설정 표는 팝업을 열기 전과 같다

### Requirement: 추천에 실패하면 기존 설정으로 학습한다
추천 계산이 오류로 실패하면(예: 데이터가 너무 짧음) 시스템은 SHALL 팝업을 열지 않고 알고리즘 설정 표의 값으로 학습을 시작하며, 실패 원인을 로그에 남긴다. 한 파라미터만 추천할 수 없는 경우 시스템은 SHALL 그 파라미터만 팝업에서 빼고 나머지는 보여 준다.

#### Scenario: 데이터가 너무 짧음
- **WHEN** 리샘플링 후 학습 데이터가 추천에 필요한 최소 길이보다 짧은 상태에서 Train을 누른다
- **THEN** 팝업 없이 설정 표의 값으로 학습이 시작되고, 추천 실패 경고가 로그에 남는다

### Requirement: max_forecast_steps는 자기상관 감쇠로 추천한다
`max_forecast_steps`를 받는 추천 대상 알고리즘은 SHALL 같은 규칙으로 `max_forecast_steps`를 추천한다(차분을 추천한 Arima·Sarima는 제외, 아래 요구사항 참고): 대상 변수의 자기상관(ACF)이 처음으로 임계값(기본 0.5) 아래로 떨어지기 직전의 시차. 검색 범위 안에서 ACF가 임계값 아래로 떨어지지 않으면 SHALL 검색 범위의 끝을 추천한다.

#### Scenario: AR(1) 과정
- **WHEN** 계수 0.95인 AR(1) 과정(ACF(h)=0.95^h)으로 추천을 계산한다
- **THEN** 추천 `max_forecast_steps`는 약 13(9 이상 19 이하)이다

### Requirement: 차분하는 ARIMA·SARIMA의 max_forecast_steps는 차분한 값의 자기상관으로 추천한다
Arima·Sarima 추천이 차분(d ≥ 1 또는 D = 1)을 포함하면 시스템은 SHALL `max_forecast_steps`를 모델이 예측하는 차분한 값의 자기상관으로 정한다: 차분한 값의 |ACF|가 95% 유의 한계 안으로 처음 들어가기 직전의 시차(최소 1). 계절 차분(D = 1)이 있거나, Arima가 계절 주기 m 이상의 AR 차수(긴 AR 차수)를 추천하면 SHALL 최소 한 주기(m)로 정한다. 근거에는 SHALL 그 시차 뒤로는 예측이 현재 수준에서 평평해진다(또는 마지막 주기를 반복한다)는 설명을 포함한다.

#### Scenario: 랜덤 워크
- **WHEN** 랜덤 워크 데이터로 Arima 추천을 계산해 d = 1이 추천된다
- **THEN** 차분한 값이 백색 잡음이므로 추천 `max_forecast_steps`는 3 이하이고, 근거에 예측이 현재 수준에서 평평해진다는 설명이 있다

#### Scenario: 계절 차분이 있는 Sarima
- **WHEN** 1시간 간격·주기 24 데이터로 Sarima 추천을 계산해 D = 1이 추천된다
- **THEN** 추천 `max_forecast_steps`는 24 이상이다

### Requirement: Arima·Sarima는 표현하지 못하는 계절 주기를 안내한다
유의한 계절 주기가 있는데 추천한 모델이 그 주기를 표현하지 못하면(Arima는 지배적 주기가 Arima 주기 상한(기본 48)보다 긴 경우, Sarima는 주기가 Sarima 상한보다 길어 계절 항을 뺀 경우) 시스템은 SHALL 팝업 노트에 그 주기와 기간, 그리고 대안을 표시한다. 대안은 SHALL 더 큰 간격으로 리샘플링하거나 그 주기를 표현하는 알고리즘을 쓰라는 안내다(Arima: LGBMForecaster/RandomForestForecaster, Sarima: LGBMForecaster/RandomForestForecaster, Prophet). 제조 데이터에서는 SARIMA 계열을 쓰지 않으므로 Arima 안내는 SHALL Sarima를 대안으로 제시하지 않는다.

#### Scenario: 1분 데이터의 1일 주기
- **WHEN** 1분 간격 데이터에서 1440(1일) 주기가 검출된 상태로 Arima 추천을 계산한다
- **THEN** 팝업 노트에 ARIMA가 1440단계(1일) 주기를 표현하지 못하며, 리샘플링하거나 LGBMForecaster 등을 쓰라는 안내가 표시되고, Sarima는 언급되지 않는다

#### Scenario: Arima 주기 상한 이하의 주기
- **WHEN** 1시간 간격·주기 24 데이터로 Arima 추천을 계산한다
- **THEN** 추천 `order`의 p는 24 이상이고(긴 AR 차수), 팝업 노트에 계절 주기 안내가 표시되지 않는다

### Requirement: 자기회귀 트리 모델의 maxlags는 지배적 계절 주기로 추천한다
LGBMForecaster, RandomForestForecaster, ExtraTreesForecaster는 SHALL 같은 규칙으로 `maxlags`를 추천한다: `max_forecast_steps`보다 긴 유의한 계절 주기 중 ACF가 임계값 이상이면서 가장 높은 주기. 그런 주기가 없으면 SHALL `max_forecast_steps` 이상이 되도록 정한다. 기존 LGBMForecaster 추천 결과는 SHALL 바뀌지 않는다.

#### Scenario: 24단계 주기 데이터
- **WHEN** 주기 24의 사인파에 잡음을 더한 데이터로 RandomForestForecaster 추천을 계산한다
- **THEN** 추천 `maxlags`는 24이다

### Requirement: Arima의 order는 후보를 Arima와 같은 방식으로 적합해 홀드아웃 오차로 고른다
Arima의 `order (p, d, q)` 추천은 SHALL 후보 차수를 만들고, 각 후보를 학습 데이터에서 최근 홀드아웃 구간을 뺀 데이터로 Arima 모델과 같은 방식(추세 항 없음, 정상성·가역성 강제 안 함)으로 적합해, 홀드아웃 구간 예측의 평균 절대 오차가 가장 작은 후보를 추천한다. 적합에 실패하거나 예측이 발산한 후보는 SHALL 고르지 않는다. 후보는 SHALL 다음과 같다.
- 짧은 차수: 단위근 검정(KPSS)으로 정한 d(최대 2)에서 정보 기준(AICc)이 가장 작은 p, q(각 0~5).
- 검정이 d = 0이면, d = 1에서 같은 탐색으로 정한 짧은 차수.
- Arima 주기 상한(기본 48) 이하의 유의한 계절 주기 m이 있으면, d ∈ {검정한 d, 1}마다 긴 AR 차수: p는 AIC가 가장 작은 AR 차수(최대 2m, 상한 60)이되 m 이상, q = 0.

홀드아웃 길이는 SHALL m이 있으면 두 주기(2m), 없으면 24단계다. 후보가 하나뿐이거나 데이터가 홀드아웃 비교에 충분히 길지 않으면 SHALL 짧은 차수를 추천한다. 근거에는 SHALL 선택한 후보의 선정 근거(검정 결과와 정보 기준 값, 또는 긴 AR 차수의 주기)와 후보별 홀드아웃 오차를 포함한다.

배경(2026-10-08 M4 Hourly 414개 시계열 벤치마크): 이전 규칙(짧은 차수만, 정상성을 강제한 AICc로 선택)은 Arima 모델이 정상성을 강제하지 않고 상수항 없이 적합하는 것과 맞지 않아, 44개 시계열에서 MASE가 10을 넘고 일부는 예측이 발산했다(중앙값 sMAPE 10.9). 이 규칙으로 바꾼 뒤 시험에서 중앙값 sMAPE 4.5, MASE가 10을 넘는 시계열은 없었다.

#### Scenario: 상수항 없이 0으로 감쇠하는 후보 배제
- **WHEN** 수준 100 주위의 정상 AR(2) 과정 데이터로 Arima 추천을 계산한다
- **THEN** d = 0 후보는 홀드아웃에서 예측이 0 쪽으로 감쇠하거나 발산해 지고, 추천 `order`의 d는 1이다

#### Scenario: 랜덤 워크
- **WHEN** 랜덤 워크(누적합 잡음) 데이터로 Arima 추천을 계산한다
- **THEN** 추천 `order`의 d는 1이다

#### Scenario: 정상 AR(2) 과정
- **WHEN** 정상 AR(2) 과정 데이터로 Arima 추천을 계산한다
- **THEN** 추천 `order`의 d는 0이고 p는 1 이상이다

### Requirement: Sarima의 계절 차수는 주기 검출과 계절 강도로 추천한다
Sarima의 `seasonal_order (P, D, Q, m)` 추천은 SHALL m을 유의한 지배적 계절 주기로 정하고, 계절 차분 D를 계절 강도가 기준(0.64)을 넘을 때 1, 아니면 0으로 정하며, P와 Q를 0~1 범위에서 정보 기준이 가장 작은 조합으로 정한다. 비계절 `order`는 SHALL Arima와 같은 규칙으로 정한다. 유의한 주기가 없거나 m이 계절 주기 상한보다 크면 SHALL `seasonal_order`를 `(0, 0, 0, 0)`으로 추천하고 근거에 그 이유를 적는다.

#### Scenario: 강한 일 주기
- **WHEN** 1시간 간격이고 주기 24가 뚜렷한 데이터로 Sarima 추천을 계산한다
- **THEN** 추천 `seasonal_order`의 m은 24이고 D는 1이다

#### Scenario: 주기가 상한보다 긺
- **WHEN** 5분 간격 데이터에서 지배적 주기가 288(1일)로 검출되고 이것이 계절 주기 상한보다 크다
- **THEN** 추천 `seasonal_order`는 `(0, 0, 0, 0)`이고, 근거에 주기가 너무 길어 계절 항을 생략했다는 설명이 있다

### Requirement: ETS 구성 요소는 분해 강도와 진폭–수준 관계로 추천한다
ETS 추천은 SHALL `seasonal_periods`를 다음 후보 중에서 정한다: 유의한 계절 주기 가운데 ACF가 기준(`ets_min_acf`, 기본 0.3) 이상이고 데이터가 두 주기 이상을 덮는 주기. 후보를 ACF가 높은 순으로 보며, 예상 학습 시간이 승인 기준(`train_confirm_seconds`, 기본 5분) 이내인 첫 주기를 추천한다. 후보가 없으면 SHALL `None`(이때 `seasonal`도 `None`)을 추천하고, 근거에 그 이유(주기가 너무 약함, 또는 학습 시간이 기준을 넘음)를 적는다. 학습 시간 때문에 뺀 주기가 있으면 SHALL 팝업 노트에 가장 강한 그 주기와 예상 학습 시간, 대안(더 큰 간격으로 리샘플링, LGBMForecaster)을 표시한다. 계절 성분의 진폭이 수준에 비례해 커지고 데이터가 모두 양수이면 `seasonal`을 `mul`, 아니면 `add`로 정한다. `trend`는 SHALL 추세 강도가 기준을 넘으면 `add`, 아니면 `None`으로 정하고, 추세가 있으면 `damped_trend`를 `True`로 정한다. `error`는 SHALL `add`로 추천한다.

배경(2026-10-09 제조 데이터 벤치마크, NAB machine temperature 21개 창과 SKAB 64개 창): 유의하지만 ACF가 0.1 미만인 주기는 예측을 평균적으로 개선하지 못했고, 그중 216~493단계 주기는 학습에 3~40분이 걸렸다. ACF 0.4 이상인 주기는 MASE를 개선했다(중앙값 약 0.18). 이 규칙으로 바꾼 뒤 machine temperature는 정확도가 같고 총 학습 시간이 40분에서 1초 미만으로, SKAB는 MASE 중앙값이 1.160에서 1.088로 줄고 총 학습 시간이 38분에서 0.5분으로 줄었다.

#### Scenario: 추세와 곱셈 계절성
- **WHEN** 선형 추세가 있고, 주기 12의 계절 진폭이 수준에 비례하는 양수 데이터로 ETS 추천을 계산한다
- **THEN** 추천은 `trend=add`, `damped_trend=True`, `seasonal=mul`, `seasonal_periods=12`이다

#### Scenario: 백색 잡음
- **WHEN** 백색 잡음 데이터로 ETS 추천을 계산한다
- **THEN** 추천은 `trend=None`, `seasonal=None`, `seasonal_periods=None`이다

#### Scenario: 약한 계절 주기
- **WHEN** NAB machine temperature 학습 데이터(5분 간격, 18,156점)처럼 가장 강한 유의 주기(1969단계)의 ACF가 0.158로 기준보다 약하다
- **THEN** `seasonal_periods`는 `None`으로 추천되고, 근거에 그 주기가 너무 약하다는 설명이 있다

#### Scenario: 학습이 오래 걸리는 긴 계절 주기
- **WHEN** 5분 간격 데이터에서 뚜렷한 주기가 288(1일)뿐이고, 그 주기로 학습하면 5분을 넘을 것으로 예상된다
- **THEN** `seasonal_periods`는 `None`으로 추천되고, 팝업 노트에 그 주기의 예상 학습 시간과 리샘플링하거나 LGBMForecaster를 쓰라는 안내가 표시된다

#### Scenario: 학습 시간 안에 드는 짧은 주기로 대체
- **WHEN** 가장 강한 주기는 학습 시간이 기준을 넘고, 그보다 약하지만 기준 이상의 ACF를 가진 짧은 주기가 있다
- **THEN** `seasonal_periods`는 그 짧은 주기로 추천되고, 근거에 예상 학습 시간과 더 강한 주기를 뺐다는 설명이 있다

### Requirement: VectorAR의 maxlags는 VAR 차수 선택 기준으로 추천한다
VectorAR의 `maxlags` 추천은 SHALL 대상 변수와 선택한 feature 변수들로 VAR 차수별 정보 기준을 계산해 BIC가 가장 작은 차수로 정하고, 근거에 AIC·BIC·HQIC가 각각 고른 차수를 보여 준다. feature 변수가 없으면 SHALL 단변량 AR 차수 선택으로 정한다.

#### Scenario: 차수 2인 VAR 과정
- **WHEN** 차수 2인 2변량 VAR 과정 데이터(대상 1개, feature 1개)로 VectorAR 추천을 계산한다
- **THEN** 추천 `maxlags`는 2이다

### Requirement: DefaultForecaster는 예측 구간과 샘플링 간격을 추천한다
DefaultForecaster 추천은 SHALL `max_forecast_steps`를 공통 자기상관 감쇠 규칙으로 정하고, `granularity`를 학습 데이터에서 추론한 샘플링 간격으로 정한다.

#### Scenario: 5분 간격 데이터
- **WHEN** 5분 간격 데이터로 DefaultForecaster 추천을 계산한다
- **THEN** 추천 `granularity`는 5분에 해당하는 값이다

### Requirement: 추천 계산은 제한된 시간 안에 끝난다
추천 계산은 SHALL 팝업을 여는 동기 요청 안에서 끝나도록, 모델 적합이 필요한 탐색에서 학습 데이터의 최근 일부(상한 길이)만 쓰고 적합 횟수를 제한한다. 사용한 데이터 길이는 SHALL 팝업에 표시한다.

#### Scenario: 긴 데이터의 Arima 추천
- **WHEN** 20,000점 이상인 데이터로 Arima 추천을 계산한다
- **THEN** 정보 기준 탐색은 최근 상한 길이만큼의 데이터로 수행되고, 팝업에 그 길이가 표시된다

### Requirement: 예상 학습 시간이 기준을 넘으면 학습 전에 승인을 받는다
서비스는 리소스가 제한된 환경에서 동작하므로, 학습을 시작하기 직전(추천 대상 알고리즘은 추천 팝업에서 Confirm한 뒤, 그 밖의 알고리즘은 Train을 누른 뒤) 시스템은 SHALL 학습에 쓸 설정과 학습 데이터(리샘플링 후 점 수, 변수 수)로 예상 학습 시간을 계산한다. 예상 시간이 기준(`ForecastModel.train_confirm_seconds`, 기본 300초 = 5분)을 넘으면 시스템은 SHALL 학습을 시작하지 않고 예상 시간과 비용의 주된 원인(예: ETS의 `seasonal_periods`)을 보여 주는 승인 팝업을 연다. 사용자가 승인하면 SHALL 그 설정으로 학습을 시작하고, 취소하면 SHALL 학습을 시작하지 않는다. 기준 이하이면 SHALL 묻지 않고 바로 학습을 시작한다.

예상 시간은 SHALL 알고리즘별로 측정한 학습 시간(비용을 좌우하는 파라미터와 데이터 길이에 따라)에 실행 환경의 속도 보정(고정된 기준 적합의 소요 시간 비)을 곱해 계산한다. 비용 모델이 없는 알고리즘이거나 예상 계산이 실패하면 SHALL 묻지 않고 학습을 시작한다.

#### Scenario: 긴 계절 주기의 ETS
- **WHEN** 5분 간격 2016점 데이터에서 ETS의 `seasonal_periods`를 493으로 Confirm한다
- **THEN** 예상 학습 시간(약 24분)이 5분을 넘으므로 승인 팝업이 열리고, 사용자가 승인하기 전에는 학습이 시작되지 않는다

#### Scenario: 승인 후 학습
- **WHEN** 승인 팝업에서 "Train anyway"를 누른다
- **THEN** 확인한 설정으로 학습이 시작된다

#### Scenario: 짧은 학습
- **WHEN** 1시간 간격 2000점 데이터에서 Arima `(4, 1, 2)`로 학습을 요청한다
- **THEN** 승인 팝업 없이 바로 학습이 시작된다
