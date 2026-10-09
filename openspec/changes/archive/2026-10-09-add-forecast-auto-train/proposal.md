## Why

Forecasting 탭에서 학습하려면 지금은 사용자가 네 단계를 거쳐야 한다. ① 파일과 변수를 선택하고, ② 알고리즘을 고르고, ③ 하이퍼파라미터를 편집하거나 추천 팝업에서 확인한 뒤, ④ 학습한다. ②와 ③은 시계열 예측 경험이 있어야 잘 할 수 있다. 이제는 `recommend_model`이 학습 데이터로 알고리즘과 하이퍼파라미터를 고를 수 있다. 499개 시계열 벤치마크에서 Merlion `ModelSelector`(기본값)와 `DefaultForecaster`보다 나았다. 그러므로 ②와 ③을 자동으로 처리하는 경로를 제공한다.

## What Changes

- Train 버튼 옆에 **Auto** 체크박스를 추가한다.
- Auto가 체크된 상태에서 Train을 누르면 다음 순서로 진행한다.
  1. 학습 데이터로 `recommend_model`을 실행해 알고리즘과 하이퍼파라미터를 고른다. 대상·특징·외생 변수를 넘기고, 예측 구간은 테스트 데이터 길이로 한다.
  2. 고른 설정으로 바로 학습하고, 테스트 지표와 그래프를 보여 준다.
  3. 파라미터 추천 확인 팝업은 열지 않는다.
- 학습이 끝나면 다음을 화면에 반영한다.
  - 알고리즘 드롭다운과 Algorithm Setting 표에 고른 알고리즘과 파라미터를 채운다.
  - 오른쪽에 **Model Selection** 카드를 새로 만든다. 이 카드는 후보별 순위, 검증 MAE, 상태(제외·실패 사유), 소요 시간, 검증 구간 정보, 안내 문구를 보여 준다.
- Auto가 체크되어 있는 동안 알고리즘 드롭다운은 비활성화한다. 사용하지 않기 때문이다.
- 고를 수 있는 후보가 없으면(모두 제외되거나 실패) 학습하지 않고 예외 팝업에 후보별 사유를 보여 준다.
- 모델 선택은 학습과 같은 백그라운드 작업 안에서 진행한다. 진행 표시줄과 Cancel 버튼이 선택 단계에서도 동작한다.

## Capabilities

### New Capabilities
- `forecast-auto-train`: Forecasting 탭에서 알고리즘과 하이퍼파라미터를 자동으로 골라 학습하는 Auto 모드. UI, 진행 흐름, 결과 표시, 실패 처리를 다룬다.

### Modified Capabilities
- `forecast-param-recommendation`: 두 요구사항에 Auto 모드 예외를 추가한다.
  - "추천 대상 알고리즘은 학습 전에 확인 팝업을 연다": Auto 모드에서는 팝업을 열지 않는다.
  - "예상 학습 시간이 기준을 넘으면 학습 전에 승인을 받는다": Auto 모드는 기준을 넘는 후보를 미리 제외하므로 승인 팝업을 거치지 않는다.

## Impact

- 코드:
  - `merlion/dashboard/pages/forecast.py`: Auto 체크박스, Model Selection 카드, 결과 Store
  - `merlion/dashboard/callbacks/forecast.py`: 학습 요청 흐름, 백그라운드 학습 콜백, 결과 반영 콜백
- 테스트: `tests/dashboard/`에 Auto 흐름 테스트를 추가한다. Dash와 분리된 `handle_*` 함수 패턴을 따른다.
- 의존: `ForecastModel.recommend_model(train_df, target_column, horizon, feature_columns=None, exog_columns=None, algorithms=None)`. 병렬 변경 `recommend-model-multivariate`가 이 시그니처를 제공한다. 이 변경은 `merlion/dashboard/models/`를 수정하지 않는다.
