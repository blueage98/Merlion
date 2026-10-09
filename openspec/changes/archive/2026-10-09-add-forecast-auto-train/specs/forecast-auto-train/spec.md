## Purpose

Forecasting 탭에서 사용자가 알고리즘 선택과 하이퍼파라미터 편집을 건너뛰고, 학습 데이터로 자동 선택한 알고리즘과 설정으로 바로 학습할 수 있게 한다.

## ADDED Requirements

### Requirement: Train 버튼 옆에 Auto 체크박스를 둔다
Forecasting 탭은 SHALL Train 버튼 옆에 "Auto" 체크박스를 보여 주며, 기본값은 체크 해제다. Auto가 체크되어 있는 동안 알고리즘 드롭다운은 SHALL 비활성화되고, 체크를 해제하면 SHALL 다시 활성화된다.

#### Scenario: 초기 화면
- **WHEN** 사용자가 Forecasting 탭을 연다
- **THEN** Train과 Cancel 버튼 옆에 체크 해제된 Auto 체크박스가 보인다

#### Scenario: Auto 체크
- **WHEN** 사용자가 Auto를 체크한다
- **THEN** 알고리즘 드롭다운이 비활성화된다

### Requirement: Auto 모드는 변수 선택 상태로 단변량·다변량을 판별한다
Auto가 체크되어 있으면 시스템은 SHALL "Select Other Features"와 "Select Exogenous Variables"의 선택 상태로 분석 유형을 판별한다.
- 두 항목이 모두 비어 있으면 **단변량**이다.
- 하나라도 선택되어 있으면 **다변량**이다. 다변량이면 특징 변수 수와 외생 변수 수도 함께 판별한다.

판별 결과는 SHALL 다음 두 곳에 표시한다.
- Train·Cancel·Auto 버튼 줄 바로 아래의 별도 줄: 예) "Auto 모드: 단변량", "Auto 모드: 다변량 (특징 2, 외생 1)". 그 아래에 SHALL 해당 유형에서 Auto 모드가 어떤 변수로 알고리즘과 하이퍼파라미터를 고르는지 한 문장으로 설명한다. 두 선택 항목이 바뀌면 SHALL 즉시 갱신한다. 좁은 왼쪽 패널에서 버튼 옆에 두면 줄바꿈되어 버튼 아래로 밀리므로, 별도 줄에 둔다.
- 학습 후 Model Selection 카드

Auto가 체크 해제되어 있으면 표시하지 않는다.

#### Scenario: 단변량 판별
- **WHEN** 대상 변수만 선택하고 Auto를 체크한다
- **THEN** 버튼 줄 아래에 "Auto 모드: 단변량"과 대상 변수만으로 고른다는 설명이 표시된다

#### Scenario: 다변량 판별
- **WHEN** Auto가 체크된 상태에서 특징 변수 2개와 외생 변수 1개를 선택한다
- **THEN** 버튼 줄 아래 표시가 "Auto 모드: 다변량 (특징 2, 외생 1)"과 특징·외생 변수를 함께 쓴다는 설명으로 바뀐다

#### Scenario: 외생 변수만 선택
- **WHEN** Auto가 체크된 상태에서 특징 변수 없이 외생 변수 1개만 선택한다
- **THEN** 버튼 줄 아래에 "Auto 모드: 다변량 (특징 0, 외생 1)"이 표시된다

### Requirement: Auto 모드는 판별한 분석 유형에 맞춰 파라미터를 추천받는다
Auto 모드의 모델 선택은 SHALL 판별한 분석 유형에 맞는 입력으로 실행한다.
- 단변량: SHALL 대상 변수만 넘긴다. 후보 구성과 파라미터 추천은 단변량 기준(`forecast-model-selection`)을 따른다.
- 다변량: SHALL 선택한 특징 변수와 외생 변수를 함께 넘긴다. 후보 구성(다변량 전용 후보 포함)과 파라미터 추천은 다변량 기준(`forecast-model-selection`)을 따른다.

Model Selection 카드는 SHALL 선택된 알고리즘의 하이퍼파라미터가 어느 분석 유형 기준으로 추천되었는지 보여 준다.

#### Scenario: 단변량 추천
- **WHEN** 단변량으로 판별된 상태에서 Auto 학습을 한다
- **THEN** 모델 선택에 특징 변수와 외생 변수가 넘어가지 않고, Model Selection 카드에 "단변량 기준 추천"이 표시된다

#### Scenario: 다변량 추천
- **WHEN** 특징 변수 3개가 선택된 다변량 상태에서 Auto 학습을 한다
- **THEN** 모델 선택에 특징 변수 3개가 넘어가 후보에 다변량 전용 후보가 포함되고, Model Selection 카드에 "다변량 기준 추천 (특징 3, 외생 0)"이 표시된다

### Requirement: Auto 모드의 Train은 모델을 자동 선택한 뒤 학습한다
Auto가 체크된 상태에서 Train을 누르면 시스템은 SHALL 다음 순서로 진행한다. 이 과정에서 알고리즘 드롭다운과 설정 표의 기존 값, 그리고 추천 확인 팝업은 쓰지 않는다.
1. 학습 데이터, 대상 변수, 특징 변수, 외생 변수로 모델 선택을 실행한다.
2. 선택된 알고리즘과 하이퍼파라미터로 학습한다.
3. 기존 학습과 같은 방식으로 학습·테스트 지표와 예측 그래프를 보여 준다.

모델 선택의 예측 구간(`horizon`)은 SHALL 테스트 데이터를 학습 데이터의 샘플링 간격으로 리샘플한 점 수로 한다. 모델 선택은 이 값을 검증 구간 길이로 상한한다. 학습된 모델의 `max_forecast_steps`는 SHALL 이 예측 구간과 같으며, 테스트 지표는 SHALL 테스트 데이터의 처음 `max_forecast_steps` 점에서 계산한다.

#### Scenario: 단변량 데이터의 Auto 학습
- **WHEN** 대상 변수만 선택하고 Auto를 체크한 뒤 Train을 누른다
- **THEN** 추천 확인 팝업 없이 모델 선택과 학습이 이어서 진행되고, 학습이 끝나면 테스트 지표와 예측 그래프가 표시된다

#### Scenario: 다변량 데이터의 Auto 학습
- **WHEN** 특징 변수와 외생 변수를 선택하고 Auto를 체크한 뒤 Train을 누른다
- **THEN** 모델 선택과 학습 모두 선택한 특징 변수와 외생 변수를 사용한다

### Requirement: Auto 학습 결과를 설정 화면과 Model Selection 카드에 보여 준다
Auto 학습이 끝나면 시스템은 SHALL 알고리즘 드롭다운 값을 선택된 알고리즘으로 바꾼다. Algorithm Setting 표에는 SHALL 선택된 하이퍼파라미터를 채운다. 오른쪽 결과 영역의 "Model Selection" 카드에는 SHALL 다음 정보를 보여 준다.
- 후보별 순위, 알고리즘, 검증 MAE, 상태(정상, 예상 학습 시간 초과로 제외, 실패와 원인), 예상 학습 시간, 검증 소요 시간
- 검증 구간 점 수와 전체 점 수, 예측 구간
- 모델 선택의 안내 문구

수동 모드(Auto 해제)로 학습하면 SHALL Model Selection 카드를 비운다.

#### Scenario: 선택 결과 반영
- **WHEN** Auto 학습에서 ExtraTreesForecaster가 선택된다
- **THEN** 알고리즘 드롭다운에 ExtraTreesForecaster가 표시되고, 설정 표에 선택된 `maxlags`와 `max_forecast_steps`가 채워지며, Model Selection 카드의 1위가 ExtraTreesForecaster다

#### Scenario: 수동 학습 후
- **WHEN** Auto 학습 뒤 Auto를 해제하고 다시 Train으로 학습한다
- **THEN** Model Selection 카드는 비어 있다

### Requirement: 선택할 후보가 없으면 학습하지 않는다
모델 선택 결과 추천 알고리즘이 없으면(모든 후보가 제외되거나 실패) 시스템은 SHALL 학습하지 않는다. 대신 예외 팝업에 후보별 상태와 사유를 보여 준다.

#### Scenario: 모든 후보 실패
- **WHEN** Auto 모드에서 모든 후보가 실패하는 데이터로 Train을 누른다
- **THEN** 학습은 진행되지 않고, 예외 팝업에 후보별 실패 사유가 표시된다

### Requirement: 모델 선택 단계도 진행 표시와 취소를 지원한다
모델 선택은 SHALL 학습과 같은 백그라운드 작업 안에서 실행한다. 그동안 SHALL 진행 표시줄이 움직이고, Train 버튼은 비활성화되며, Cancel 버튼으로 중단할 수 있다.

#### Scenario: 선택 중 취소
- **WHEN** Auto 모드에서 모델 선택이 진행 중일 때 Cancel을 누른다
- **THEN** 선택과 학습이 중단되고 Train 버튼이 다시 활성화된다
