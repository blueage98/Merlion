# 세션 노트 (마지막 업데이트: 2026-10-05)

## 남은 작업

1. **알고리즘별 추천 파라미터 함수 만들기**
   - 목표: `recommend_lgbm_params`처럼, 알고리즘마다 최적화가 필요한 파라미터의 추천값을 제시하는 함수를 각각 만들기.
   - 시작점: `merlion/dashboard/models/forecast.py`의 `ForecastModel.algorithms`(DefaultForecaster, Arima, LGBMForecaster, ETS, AutoETS, Prophet, AutoProphet, Sarima, VectorAR, RandomForestForecaster, ExtraTreesForecaster). 각 config 클래스(`ModelFactory.get_model_class(name).config_class`)에서 최적화가 필요한 파라미터를 정리합니다.
   - 참고 패턴: `ForecastModel.recommend_lgbm_params()`(ACF 기반 추천, 근거 통계도 함께 반환), `callbacks/forecast.py`의 `confirm_train_settings` 팝업, `tuned_algorithms` 목록.
   - 설계 방향:
     - 추천 함수를 `{알고리즘: 추천 함수}` 등록 테이블로 일반화하고, `_create_recommendation_content()`가 파라미터 목록을 받아 표를 그리게 바꿉니다.
     - RandomForest와 ExtraTrees는 LGBM과 같은 sklearn 기반이라 같은 로직을 재사용할 수 있습니다.
     - Arima/Sarima/ETS는 계절 주기와 차수가 핵심입니다. Auto* 계열은 이미 내부에서 자동 탐색합니다(`merlion/models/automl/`).
   - 검증: `python -m pytest tests/dashboard -q`에 알고리즘별 테스트를 추가하고, Forecasting 탭에서 Train → 팝업 → Confirm 흐름을 확인합니다. 수동 확인용 데이터는 `example.csv`(1분, 불규칙 간격)와 `machine_temperature_system_failure.csv`(5분)입니다.

## 보류 중인 문제

- **`.claude/settings.json` 권한 변경 (커밋 여부 결정 필요)**: 커밋된 버전은 `deny`에 `Bash(git commit:*)`와 `Bash(git push:*)`가 있어서 Claude의 커밋과 push를 막습니다. 작업 트리에서는 2026-10-05에 이 두 줄이 빠져 있지만(Claude가 바꾼 것이 아님), 팀 전체 정책이라 커밋하지 않았습니다. Claude의 커밋/push 권한은 개인용 `.claude/settings.local.json`(gitignore 처리)에 따로 허용해 두었습니다.

- **`test_moving_average` 실패**: `tests/transform/test_moving_average.py::test_exponential_moving_average_ci`가 `KeyError: 1`로 실패합니다. 원래부터 있던 문제이고, pandas 3의 정수 라벨 인덱싱(`series[1]`) 문제로 추정합니다.
- **Prophet 학습 오류**: `AttributeError: 'Prophet' object has no attribute 'stan_backend'`. 2026-10-05에 사용자가 처리하지 않기로 했습니다.
