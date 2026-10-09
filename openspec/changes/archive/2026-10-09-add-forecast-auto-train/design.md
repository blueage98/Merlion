## Context

현재 학습 흐름([callbacks/forecast.py](../../../merlion/dashboard/callbacks/forecast.py))은 다음과 같다.

1. `confirm_train_settings`: Train을 누르면 추천 대상 알고리즘이면 팝업을 연다. 그렇지 않으면 `forecasting-train-request` Store에 `{"time", "overrides"}`를 쓴다.
2. `gate_long_training`: 요청을 받아 학습 시간을 추정하고, 기준을 넘으면 승인 팝업을 연다. 아니면 `forecasting-train-trigger`에 쓴다.
3. `click_train_test`: trigger를 받아 학습한다. 이 콜백은 백그라운드 콜백이고, 진행 표시줄과 Cancel, Train 비활성화를 지원한다.

Dash 로직은 `handle_train_settings`, `handle_train_request`처럼 테스트 가능한 순수 함수로 분리하는 패턴을 쓴다. 이 변경이 쓰는 모델 선택 인터페이스는 병렬 변경 `recommend-model-multivariate`의 계약을 따른다.
```python
ForecastModel.recommend_model(train_df, target_column, horizon,
                              feature_columns=None, exog_columns=None, algorithms=None) -> ModelRecommendation
# ModelRecommendation: algorithm, params, candidates[ModelCandidate(algorithm, params, valid_mae,
#                      estimated_seconds, eval_seconds, status)], n_valid, n_points, notes
```

## Goals / Non-Goals

**Goals:**
- 기존 3단계 흐름(request → gate → trigger)을 유지하면서, 요청에 `auto` 플래그를 실어 분기한다.
- 모델 선택은 백그라운드 학습 콜백 안에서 실행한다. 이렇게 하면 진행 표시, 취소, 버튼 비활성화를 그대로 쓸 수 있다.
- 새 분기 로직도 Dash와 분리된 함수로 만들어 단위 테스트한다.

**Non-Goals:**
- 후보 목록이나 검증 방식을 UI에서 설정하는 기능은 넣지 않는다.
- 선택 결과를 저장하는 기능은 넣지 않는다. 모델 저장은 기존 `save_model`이 그대로 한다.
- `merlion/dashboard/models/`는 수정하지 않는다. 병렬 변경의 영역이다.

## Decisions

1. **요청 흐름에 `auto` 플래그를 싣는다.**
   - `confirm_train_settings`에 `State("forecasting-auto-train", "value")`를 추가한다. `handle_train_settings`는 Auto이고 Train을 눌렀으면 팝업 없이 `{"time", "overrides": {}, "auto": True}`를 반환한다.
   - `handle_train_request`는 `request.get("auto")`이면 추정 없이 바로 trigger로 넘긴다.
   - 대안은 Auto 전용 버튼과 별도 콜백 체인을 만드는 것이다. 하지만 같은 Store와 trigger를 공유하는 편이 진행 표시와 취소를 중복 없이 재사용한다.
2. **모델 선택은 `click_train_test` 안에서 실행한다.**
   - trigger에 `auto`가 있으면 학습·테스트 데이터를 불러오고, `horizon`을 계산한 뒤, `ForecastModel.recommend_model`을 호출한다. 이 부분은 새 함수 `run_auto_selection(train_df, test_df, target, features, exog, recommend=...)`로 분리해 테스트한다.
   - 추천 알고리즘이 없으면 후보별 상태를 담은 예외를 던져 기존 예외 팝업 경로로 보여 준다.
   - 진행 표시는 선택 단계 0→2, 학습 단계 2→10으로 나눈다. 기존 `train`은 2부터 진행한다.
3. **`horizon`은 테스트 데이터를 학습 granularity로 리샘플한 점 수로 계산한다.**
   - 학습 데이터로 `TemporalResample`을 학습하고, 테스트 데이터의 대상 열에 적용해 점 수를 센다. 불규칙 간격 데이터에서도 "스텝 수"가 학습 모델과 일치한다.
   - 대안인 테스트 행 수는 원본이 불규칙하면 스텝 수와 다르다.
   - `recommend_model`이 `horizon`을 검증 구간 길이로 상한한다.
4. **결과는 새 Store `forecasting-auto-result`로 전달한다.**
   - 백그라운드 콜백은 이 Store 하나에 `{algorithm, params, candidates, n_valid, n_points, horizon, notes}`를 쓴다. 수동 학습이면 `None`을 쓴다.
   - 일반 콜백들이 이 Store를 받아 화면을 갱신한다.
     - (a) 알고리즘 드롭다운 값을 바꾼다.
     - (b) Model Selection 카드를 그린다.
   - 대안은 백그라운드 콜백이 드롭다운과 설정 표를 직접 `allow_duplicate`로 갱신하는 것이다. 이 방식에는 문제가 있다. 드롭다운 값이 바뀌면 기존 `select_algorithm`이 설정 표를 기본값으로 다시 그려서 두 콜백이 경쟁한다.
5. **설정 표 갱신은 `select_algorithm`이 Store를 참조해 처리한다.**
   - `select_algorithm`에 `State("forecasting-auto-result", "data")`를 추가한다. 선택된 알고리즘이 Store의 알고리즘과 같으면 기본 설정에 Store의 params를 덮어써서 표를 만든다. 확인 팝업 Confirm과 같은 `create_param_table({name: {"default": value}})` 형식이다.
   - 이렇게 하면 드롭다운 변경으로 생기는 한 번의 갱신만으로 표가 최종값이 되어 경쟁이 없다.
   - 사용자가 Auto 결과 뒤에 다른 알고리즘을 고르면 기본값 표가 나온다.
6. **Auto 체크박스는 `dbc.Checkbox(id="forecasting-auto-train", label="Auto", value=False)`로 만들어 Train과 Cancel 옆 같은 줄에 둔다.**
   - 체크 상태가 바뀌면 `forecasting-select-algorithm.disabled`를 갱신하는 콜백을 둔다.
7. **분석 유형은 callbacks에서 판별한다.**
   - `analysis_mode(feature_cols, exog_cols) -> {"kind": "univariate"|"multivariate", "n_features", "n_exog"}` 순수 함수를 둔다.
   - 쓰는 곳
     - (a) Auto 옆 라벨(`forecasting-auto-mode`) 갱신 콜백: Input은 Auto, features, exog
     - (b) `run_auto_selection`: 단변량이면 `feature_columns`와 `exog_columns`를 넘기지 않고, 다변량이면 선택 목록을 넘긴다.
     - (c) 결과 Store와 Model Selection 카드: "단변량 기준 추천" 또는 "다변량 기준 추천 (특징 n, 외생 m)"
   - 유형별 파라미터 추천의 실제 계산은 `recommend_model` 쪽 책임이다(`recommend-model-multivariate`의 "하이퍼파라미터 추천은 분석 유형에 따라 다르게 계산한다"). 이 변경은 판별과 입력 전달만 한다.
   - 대안은 `ModelRecommendation`에 유형 필드를 추가하는 것이다. 하지만 이 방식은 병렬 계약을 바꾼다. 유형은 입력만으로 결정되므로 UI 쪽에서 계산한다.

8. **Model Selection 카드는 기존 결과 카드(`result_table_card`)와 같은 스타일로 만든다.**
   - 위치는 Forecasting Results 위다.
   - 표의 열은 순위, 알고리즘, 검증 MAE, 상태, 예상 학습 시간, 검증 시간이다. 표 아래에 검증 구간 설명과 notes를 둔다.
   - 비어 있을 때는 "Auto 모드로 학습하면 후보 비교 결과가 표시됩니다" 안내를 보여 준다.

## Risks / Trade-offs

- [모델 선택이 데이터가 크면 수 분 걸림(후보 6~7개 검증)] → 백그라운드 실행, 진행 표시, Cancel로 대응한다. 각 후보는 학습 시간 상한으로 걸러진다.
- [병렬 변경이 아직 반영되지 않으면 `feature_columns`, `exog_columns` 키워드 호출이 실패함] → 단위 테스트는 `recommend` 함수를 주입해 독립적으로 검증한다. 통합 확인(수동 브라우저 테스트)은 두 변경을 모두 반영한 뒤 한다.
- [백그라운드 콜백의 출력 추가로 기존 콜백 시그니처가 바뀜] → `tests/dashboard/` 전체를 통과하는지 확인한다.
- [`select_algorithm`이 Store를 참조하면, 사용자가 같은 알고리즘을 다시 고를 때도 Auto 값이 다시 채워짐] → 의도된 동작으로 둔다(마지막 Auto 결과 유지). 수동 학습하면 Store가 `None`이 되어 해제된다.
