## MODIFIED Requirements

### Requirement: 추천 대상 알고리즘은 학습 전에 확인 팝업을 연다
Train 버튼을 누르면, 선택한 알고리즘이 추천 대상(LGBMForecaster, RandomForestForecaster, ExtraTreesForecaster, Arima, Sarima, ETS, VectorAR, DefaultForecaster)인 경우 시스템은 SHALL 학습을 시작하지 않고 추천 설정 팝업을 연다. 추천 대상이 아닌 알고리즘(Prophet, AutoETS, AutoProphet 등)은 SHALL 팝업 없이 바로 학습을 시작한다. Auto 체크박스가 체크되어 있으면 시스템은 SHALL 선택한 알고리즘과 관계없이 팝업을 열지 않고 자동 선택 학습(`forecast-auto-train`)을 진행한다.

추천 대상은 추천값이 최적화 없이 정한 기본값보다 나은 알고리즘으로 한정한다. Prophet은 2026-10-09 벤치마크(M4 Hourly 414개, 제조 데이터 85개 창)에서 추천값이 Prophet 자체의 `auto` 계절성보다 유의하게 나빴고(M4: 개선 183, 악화 231, Wilcoxon p = 0.0003), 후보 설정을 홀드아웃으로 고르도록 고쳐도 제조 데이터에서는 개선이 없어(machine temperature 9:9, SKAB 64개 모두 동일) 추천 대상에서 뺐다.

#### Scenario: 추천 대상 알고리즘 선택 후 Train
- **WHEN** 사용자가 알고리즘으로 Sarima를 선택하고 Train을 누른다
- **THEN** 추천 설정 팝업이 열리고, 사용자가 Confirm을 누르기 전에는 학습이 시작되지 않는다

#### Scenario: 자동 탐색 알고리즘 선택 후 Train
- **WHEN** 사용자가 알고리즘으로 AutoETS를 선택하고 Train을 누른다
- **THEN** 팝업 없이 알고리즘 설정 표의 값으로 바로 학습이 시작된다

#### Scenario: Auto 모드에서 Train
- **WHEN** 알고리즘 드롭다운에 Sarima가 선택된 채로 Auto를 체크하고 Train을 누른다
- **THEN** 추천 설정 팝업은 열리지 않고 자동 선택 학습이 시작된다

### Requirement: 예상 학습 시간이 기준을 넘으면 학습 전에 승인을 받는다
서비스는 리소스가 제한된 환경에서 동작하므로, 학습을 시작하기 직전(추천 대상 알고리즘은 추천 팝업에서 Confirm한 뒤, 그 밖의 알고리즘은 Train을 누른 뒤) 시스템은 SHALL 학습에 쓸 설정과 학습 데이터(리샘플링 후 점 수, 변수 수)로 예상 학습 시간을 계산한다. 예상 시간이 기준(`ForecastModel.train_confirm_seconds`, 기본 300초 = 5분)을 넘으면 시스템은 SHALL 학습을 시작하지 않고 예상 시간과 비용의 주된 원인(예: ETS의 `seasonal_periods`)을 보여 주는 승인 팝업을 연다. 사용자가 승인하면 SHALL 그 설정으로 학습을 시작하고, 취소하면 SHALL 학습을 시작하지 않는다. 기준 이하이면 SHALL 묻지 않고 바로 학습을 시작한다.

예상 시간은 SHALL 알고리즘별로 측정한 학습 시간(비용을 좌우하는 파라미터와 데이터 길이에 따라)에 실행 환경의 속도 보정(고정된 기준 적합의 소요 시간 비)을 곱해 계산한다. 비용 모델이 없는 알고리즘이거나 예상 계산이 실패하면 SHALL 묻지 않고 학습을 시작한다.

Auto 모드(`forecast-auto-train`)의 학습 요청은 SHALL 승인 팝업 없이 진행한다. 자동 선택이 예상 학습 시간이 기준을 넘는 후보를 이미 제외하므로, 선택된 설정은 기준 이하다.

#### Scenario: 긴 계절 주기의 ETS
- **WHEN** 5분 간격 2016점 데이터에서 ETS의 `seasonal_periods`를 493으로 Confirm한다
- **THEN** 예상 학습 시간(약 24분)이 5분을 넘으므로 승인 팝업이 열리고, 사용자가 승인하기 전에는 학습이 시작되지 않는다

#### Scenario: 승인 후 학습
- **WHEN** 승인 팝업에서 "Train anyway"를 누른다
- **THEN** 확인한 설정으로 학습이 시작된다

#### Scenario: 짧은 학습
- **WHEN** 1시간 간격 2000점 데이터에서 Arima `(4, 1, 2)`로 학습을 요청한다
- **THEN** 승인 팝업 없이 바로 학습이 시작된다

#### Scenario: Auto 모드 학습 요청
- **WHEN** Auto 모드로 Train을 누른다
- **THEN** 승인 팝업 없이 모델 선택이 시작된다
