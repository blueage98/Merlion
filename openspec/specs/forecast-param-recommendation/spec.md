# forecast-param-recommendation Specification

## Purpose

대시보드 Forecasting 탭에서 학습을 시작하기 전에, 선택한 알고리즘의 핵심 하이퍼파라미터 추천값을 학습 데이터로부터 계산해 근거와 함께 보여 주고, 사용자가 확인하거나 수정한 값으로 학습하게 합니다.

## Requirements

### Requirement: 추천 대상 알고리즘은 학습 전에 확인 팝업을 연다
Train 버튼을 누르면, 선택한 알고리즘이 추천 대상(LGBMForecaster, RandomForestForecaster, ExtraTreesForecaster, Arima, Sarima, ETS, Prophet, VectorAR, DefaultForecaster)인 경우 시스템은 SHALL 학습을 시작하지 않고 추천 설정 팝업을 연다. 추천 대상이 아닌 알고리즘(AutoETS, AutoProphet 등)은 SHALL 팝업 없이 바로 학습을 시작한다.

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
Arima·Sarima 추천이 차분(d ≥ 1 또는 D = 1)을 포함하면 시스템은 SHALL `max_forecast_steps`를 모델이 예측하는 차분한 값의 자기상관으로 정한다: 차분한 값의 |ACF|가 95% 유의 한계 안으로 처음 들어가기 직전의 시차(최소 1). 계절 차분(D = 1)이 있으면 SHALL 최소 한 주기(m)로 정한다. 근거에는 SHALL 그 시차 뒤로는 예측이 현재 수준에서 평평해진다(또는 마지막 주기를 반복한다)는 설명을 포함한다.

#### Scenario: 랜덤 워크
- **WHEN** 랜덤 워크 데이터로 Arima 추천을 계산해 d = 1이 추천된다
- **THEN** 차분한 값이 백색 잡음이므로 추천 `max_forecast_steps`는 3 이하이고, 근거에 예측이 현재 수준에서 평평해진다는 설명이 있다

#### Scenario: 계절 차분이 있는 Sarima
- **WHEN** 1시간 간격·주기 24 데이터로 Sarima 추천을 계산해 D = 1이 추천된다
- **THEN** 추천 `max_forecast_steps`는 24 이상이다

### Requirement: Arima·Sarima는 표현하지 못하는 계절 주기를 안내한다
유의한 계절 주기가 있는데 추천한 모델이 그 주기를 표현하지 못하면(Arima, 또는 주기가 상한보다 길어 계절 항을 뺀 Sarima) 시스템은 SHALL 팝업 노트에 그 주기와 기간, 그리고 대안을 표시한다. 주기가 Sarima 상한 이하이면 대안은 SHALL Sarima이고, 상한보다 길면 SHALL 더 큰 간격으로 리샘플링하거나 그 주기를 표현하는 알고리즘(LGBMForecaster/RandomForestForecaster, Prophet)을 쓰라는 안내다.

#### Scenario: 1분 데이터의 1일 주기
- **WHEN** 1분 간격 데이터에서 1440(1일) 주기가 검출된 상태로 Arima 추천을 계산한다
- **THEN** 팝업 노트에 ARIMA가 1440단계(1일) 주기를 표현하지 못하며, 리샘플링하거나 LGBMForecaster·Prophet 등을 쓰라는 안내가 표시된다

#### Scenario: Sarima로 표현 가능한 주기
- **WHEN** 1시간 간격·주기 24 데이터로 Arima 추천을 계산한다
- **THEN** 팝업 노트에 Sarima를 쓰라는 안내가 표시된다

### Requirement: 자기회귀 트리 모델의 maxlags는 지배적 계절 주기로 추천한다
LGBMForecaster, RandomForestForecaster, ExtraTreesForecaster는 SHALL 같은 규칙으로 `maxlags`를 추천한다: `max_forecast_steps`보다 긴 유의한 계절 주기 중 ACF가 임계값 이상이면서 가장 높은 주기. 그런 주기가 없으면 SHALL `max_forecast_steps` 이상이 되도록 정한다. 기존 LGBMForecaster 추천 결과는 SHALL 바뀌지 않는다.

#### Scenario: 24단계 주기 데이터
- **WHEN** 주기 24의 사인파에 잡음을 더한 데이터로 RandomForestForecaster 추천을 계산한다
- **THEN** 추천 `maxlags`는 24이다

### Requirement: Arima의 order는 단위근 검정과 정보 기준으로 추천한다
Arima의 `order (p, d, q)` 추천은 SHALL 차분 차수 d를 단위근 검정으로 정하고(최대 2), 그 d에서 p와 q를 정보 기준(AICc)이 가장 작은 조합으로 정한다. p와 q의 탐색 범위는 SHALL 각각 0~5로 제한한다. 근거에는 SHALL 검정 결과와 선택한 조합의 정보 기준 값을 포함한다.

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
ETS 추천은 SHALL `seasonal_periods`를 유의한 지배적 계절 주기로 정하고(없으면 `None`, 이때 `seasonal`도 `None`), 계절 성분의 진폭이 수준에 비례해 커지고 데이터가 모두 양수이면 `seasonal`을 `mul`, 아니면 `add`로 정한다. `trend`는 SHALL 추세 강도가 기준을 넘으면 `add`, 아니면 `None`으로 정하고, 추세가 있으면 `damped_trend`를 `True`로 정한다. `error`는 SHALL `add`로 추천한다. 추천 주기가 학습 시간 경고 기준보다 길면 SHALL 그 주기를 그대로 추천하되 경고를 함께 보여 준다.

#### Scenario: 추세와 곱셈 계절성
- **WHEN** 선형 추세가 있고, 주기 12의 계절 진폭이 수준에 비례하는 양수 데이터로 ETS 추천을 계산한다
- **THEN** 추천은 `trend=add`, `damped_trend=True`, `seasonal=mul`, `seasonal_periods=12`이다

#### Scenario: 백색 잡음
- **WHEN** 백색 잡음 데이터로 ETS 추천을 계산한다
- **THEN** 추천은 `trend=None`, `seasonal=None`, `seasonal_periods=None`이다

#### Scenario: 학습이 오래 걸리는 긴 계절 주기
- **WHEN** 5분 간격 데이터에서 지배적 주기가 288(1일)로 검출되어, 학습 시간 경고 기준(기본 48단계)보다 길다
- **THEN** `seasonal_periods`는 그 주기로 추천되고, 그 근거와 팝업 노트에 ETS 학습이 매우 오래 걸릴 수 있으며 더 큰 간격으로 리샘플링하거나 주기를 줄이거나 `seasonal`을 `None`으로 바꾸라는 경고가 표시된다

### Requirement: Prophet 계절성은 달력 주기의 관측 가능성과 유의성으로 추천한다
Prophet 추천은 SHALL 연·주·일 계절성 각각을 다음 조건을 모두 만족할 때만 `True`, 아니면 `False`로 정한다: 학습 데이터가 그 주기의 2배 이상 길다, 샘플링 간격이 그 주기의 절반보다 짧다, 그 주기 근처에서 자기상관이 유의하다. `seasonality_mode`는 SHALL ETS와 같은 진폭–수준 기준으로 `multiplicative` 또는 `additive`로 정한다.

#### Scenario: 수 주 분량의 시간 단위 데이터
- **WHEN** 1시간 간격, 8주 분량이고 일 주기가 뚜렷한 데이터로 Prophet 추천을 계산한다
- **THEN** `daily_seasonality=True`, `yearly_seasonality=False`이다

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
