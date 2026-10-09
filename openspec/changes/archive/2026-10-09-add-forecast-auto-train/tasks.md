## 1. 레이아웃

- [x] 1.1 `pages/forecast.py`에 다음을 추가한다: Train·Cancel 옆의 `dbc.Checkbox(id="forecasting-auto-train", label="Auto")`, `dcc.Store(id="forecasting-auto-result")`, 오른쪽 열의 Model Selection 카드(`forecasting-model-selection`). 검증: 레이아웃 생성 테스트에서 세 id가 존재
- [x] 1.2 Model Selection 카드 내용을 만드는 `create_model_selection_content(result)`를 추가한다. 빈 상태는 안내 문구, 결과가 있으면 순위 표·검증 구간·notes를 보여 준다. 검증: 빈 결과와 후보 3개 결과로 만든 컴포넌트의 행 수와 텍스트를 확인하는 테스트

## 2. 요청 흐름

- [x] 2.1 `handle_train_settings`에 `auto` 인자를 추가한다. Auto이면 Train에서 팝업 없이 `{"auto": True}` 요청을 반환한다. `confirm_train_settings`에 State를 연결한다. 검증: Sarima + Auto에서 팝업이 열리지 않는지 확인하는 테스트, 기존 `test_recommend_popup.py` 통과
- [x] 2.2 `handle_train_request`가 `auto` 요청을 추정 없이 trigger로 넘기게 한다. 검증: estimate 함수가 호출되지 않는지 확인하는 테스트, 기존 `test_long_training_gate.py` 통과
- [x] 2.3 Auto 체크 상태에 따라 알고리즘 드롭다운 `disabled`를 바꾸는 콜백을 추가한다. 검증: 핸들러 함수 테스트
- [x] 2.4 `analysis_mode(feature_cols, exog_cols)`를 구현한다. Auto 옆 라벨 `forecasting-auto-mode`(레이아웃에 추가)를 Auto, features, exog 변경에 따라 갱신하는 콜백도 추가한다. Auto 해제 시에는 빈 값이다. 검증: 단변량, 특징만, 외생만, 둘 다, Auto 해제 경우의 라벨 텍스트 테스트

## 3. 자동 선택과 학습

- [x] 3.1 `run_auto_selection(train_df, test_df, target, features, exog, recommend=...)`을 구현한다.
  - 리샘플 기준 `horizon`을 계산한다.
  - `analysis_mode`로 단변량과 다변량을 판별해, 유형에 맞는 인자로 `recommend_model`을 호출한다.
  - 결과를 직렬화 가능한 dict로 만든다. 분석 유형도 포함한다.
  - 추천 알고리즘이 없으면 후보별 사유를 담은 예외를 던진다.

  검증: 주입한 가짜 `recommend`로 다음을 확인하는 테스트
  - horizon 계산(불규칙 간격 포함)
  - 단변량이면 `feature_columns`와 `exog_columns` 없이 호출되고, 다변량이면 선택 목록이 전달되는지
  - 결과 dict
  - 예외 메시지
- [x] 3.2 `click_train_test`에서 trigger의 `auto`를 분기한다. 선택(진행 0→2) 뒤 선택된 알고리즘과 params로 학습한다. 출력에 `forecasting-auto-result`를 추가하고, 수동 학습이면 `None`을 쓴다. 검증: `python -m pytest tests/dashboard -q` 통과
- [x] 3.3 Store를 받아 알고리즘 드롭다운 값과 Model Selection 카드를 갱신하는 콜백을 추가한다. `select_algorithm`이 Store의 params로 설정 표를 채우게 한다. 검증: `select_algorithm` 핸들러 로직 테스트(같은 알고리즘이면 params 반영, 다르면 기본값)

## 4. 통합 확인

- [x] 4.1 두 변경(`recommend-model-multivariate` 포함)을 반영한 뒤 대시보드를 띄워 수동으로 확인한다(run_merlion 스킬). 데이터는 `example.csv`(단변량)와 SKAB(특징 변수 포함)이다. 확인할 것: Auto 체크 → Train → 진행 표시 → 결과 그래프·지표·Model Selection 카드 → 드롭다운·설정 표 반영. 검증: 스크린샷 또는 확인 결과 보고
- [x] 4.2 선택 중 Cancel과 모든 후보 실패 시 예외 팝업을 확인한다. 검증: 수동 확인 결과 보고
