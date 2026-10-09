# 세션 노트 (마지막 업데이트: 2026-10-09)

## 다음 세션 작업
1. **시계열 예측 엔진 구성** (사용자 요청)
   - 요구사항
     1. 외부 데이터 소스가 지정되면, 그 소스에 쌓인 데이터를 가져와 학습하고 모델과 하이퍼파라미터를 고른다.
     2. 학습할 때 몇 개의 데이터 포인트를 예측할지 입력받는 것이 핵심이다. 예: 1초마다 데이터가 쌓이고 30초 뒤까지의 추세를 보고 싶다면, 30초마다 예측 데이터를 생성한다.
     3. 그래프에는 실제 데이터와 예측 데이터를 함께 그린다. 예측은 30초 단위로 생성되므로, 예측선이 실제 데이터보다 30초 앞서 나가게 그려진다.
   - 시작점
     - 모델 선택: `ForecastModel.recommend_model(train_df, target, horizon, feature_columns, exog_columns)` (`merlion/dashboard/models/forecast.py`, `model_select.py`). 예측할 포인트 수가 그대로 `horizon`이 되고, 학습된 모델의 `max_forecast_steps`로 쓰인다.
     - 열 배치: `ForecastModel.arrange_columns`
     - 학습과 테스트 구간 자르기: `ForecastModel.train`, `_forecast_horizon_end`
     - 재학습 없이 새 데이터로 예측: `model.forecast(time_stamps, time_series_prev=최근 데이터)`
     - 학습·예측 주기 시뮬레이션: `merlion/evaluate/forecast.py`의 `ForecastEvaluator`(`cadence`, `horizon`, `retrain_freq`)
     - Auto 학습 흐름 참고: `run_auto_selection` (`merlion/dashboard/callbacks/forecast.py`)
     - 실제·예측 그래프: `model.plot_forecast_plotly`
   - 절차: OpenSpec 변경으로 계획을 먼저 세운다(`/opsx:propose`). 아래 "결정 필요"의 엔진 관련 질문을 먼저 사용자에게 확인한다.
   - 검증
     - 1초 간격 데이터(`data/manufacturing/skab_anomaly_free.csv`, `;` 구분자)로 데이터 유입을 흉내 내고, 30초 주기로 예측이 생성되어 그래프에서 30초 앞서 나가는지 확인한다.
     - `python -m pytest tests/dashboard -q`
2. **(완료) Auto 학습·다변량 선택**: 두 OpenSpec 변경은 2026-10-09에 완료되어 `openspec/changes/archive/`로 보관했고, 커밋도 했다(`d57035a`, `551ad62`). 2026-10-09에 main으로 fast-forward 병합하고 push했다(`origin/main` = `551ad62`).
   - 메인 스펙 변경: `forecast-model-selection`, `forecast-auto-train` 새로 만듦, `forecast-param-recommendation` 수정
   - 브라우저 자동 확인 방법: playwright를 `pip install --target <scratchpad>/pw`로 설치하고 `channel="msedge"`로 실행했다. 모델 선택 중 Cancel, 모든 후보 실패 팝업, 진행 표시줄(선택 중 1, 학습 중 2→8→10)을 이 방법으로 확인했다.

3. **(완료) `sample-data-selection` 스펙 검증 실패 수정**: 2026-10-09에 메인 스펙의 `## Purpose` TBD 문구를 실제 목적으로 바꿨다. `openspec validate --all --strict` 4개 모두 통과.

## 결정 필요
- **예측 엔진 설계** (1번 작업 전에 확인)
  - 데이터 소스 종류: DB, CSV 폴더, REST, 메시지 큐 중 무엇인지
  - 실행 위치: 대시보드 새 탭인지, 별도 서비스·스크립트인지
  - 재학습 주기: 모델 선택을 언제 다시 할지(고정 주기, 성능 저하 시)
  - 예측 생성 주기와 예측 구간의 관계: 둘을 같게 둘지(예: 30초마다 30초 앞 예측), 따로 설정할지
- **예측 구간 1스텝 버그**: `horizon=1`이면 `ForecastEvaluator`가 예측 구간을 최소 2점으로 넓혀 모든 후보가 실패한다. 원래부터 있던 버그다. 예측 엔진에서 짧은 구간을 쓴다면 고쳐야 한다. 제안: 검증할 때만 `max_forecast_steps`를 최소 2로 둔다.
- **설정 표 높이**: Algorithm Setting 표 영역이 작아 Auto 결과 7행 중 2행만 보인다. 넓힐지 정해야 한다.
- **VectorAR 기본 후보 유지 여부**: 다변량 벤치마크에서 VectorAR이 빠져도 손해가 없었고, 넣어도 이득이 없었다. 현재는 설계 기준에 따라 유지 중이다.
- **(선택) Spark에서 시계열별 모델 자동 선택**: `merlion/spark/pandas_udf.py`의 `forecast` UDF 안에서 `recommend_model`을 호출하는 확장. 시계열당 비용이 후보 수만큼 늘어난다.

## 현재 상태
- **커밋**: 2026-10-09에 `d57035a`(다변량 모델 선택과 테스트 구간 버그 수정), 그다음 커밋(Auto 학습 모드, "Add an Auto mode ...")로 커밋했다. `forecast-benchmark-arima-rec` 브랜치를 main에 fast-forward 병합해 push했다. 현재 브랜치는 `main`이다.
- **커밋되지 않은 파일**: 없음. 2026-10-09에 `data/M4/`, `data/manufacturing/`, `results/`, `.github/` 지침 파일도 커밋했다. main은 `origin/main`보다 앞서 있고 아직 push하지 않았다.
- **실행 중인 서버**: 없음
- **환경**: Python 3.13.14, dash 2.18.0, numpy 1.26.4, pandas 2.3.3, Java 1.8. pyspark는 설치되어 있지 않다.
- **최근 테스트 결과**: `python -m pytest tests/dashboard -q` → 151 passed (2026-10-09)

## 이번 세션에서 한 일
- **AutoETS와 ETS 추천 비교**: ETS 추천이 세 데이터셋 모두에서 더 정확하고 3~30배 빨라, 모델 추천 후보에서 AutoETS를 뺐다.
  - 이유: AutoETS는 기본 lag 40 이하에서만 주기를 찾는다.
- **Merlion 원래 기능과 비교** (499개 시계열): `recommend_model`의 MASE 중앙값이 0.747로 가장 좋았다. 원래 기능인 `ModelSelector`(기본 파라미터)는 0.813, `DefaultForecaster`는 0.916이었다.
  - 차이는 주로 추천 하이퍼파라미터에서 나온다.
- **다변량 모델 선택**: 후보를 학습과 같은 열 구성·외생 변수 처리로 검증하도록 바꿨다. 특징 변수가 있으면 VectorAR을 후보에 넣는다.
  - 다변량 88개 시계열에서 MASE 중앙값이 1.200에서 0.831로 좋아졌다(p = 1e-5).
- **Auto 학습 모드**
  - Train 옆에 Auto 체크박스를 두고, 단변량·다변량 판별 결과와 설명을 버튼 줄 아래에 표시한다.
  - 학습이 끝나면 Model Selection 카드, 알고리즘 드롭다운, 설정 표에 결과를 반영한다.
- **버그 수정**: 불규칙 간격 데이터에서 테스트 구간을 행 개수로 자르던 문제를 시간 기준으로 고쳤다.

## 알게 된 사실 / 주의사항
- 다변량 데이터에서 대상 변수만으로 검증하면 트리 모델이 자주 뽑힌다. 그런데 실제로는 모든 변수로 학습되기 때문에 성능이 크게 나빠진다. 검증 조건을 학습과 맞춰야 한다.
- `recommend_model`은 `horizon`을 검증 구간(학습 데이터의 20%) 길이로 제한한다. 그래서 테스트 데이터가 길면 `max_forecast_steps`가 그보다 짧아진다.
- SKAB 원본은 `;` 구분자이고 중간에 2초 간격이 섞여 있다. 대시보드에 올리려면 `,` 구분자로 바꿔야 한다. 변환한 파일은 `C:\Users\lsed\merlion\data\skab_multivariate.csv`에 업로드되어 있다.
- 대시보드 데이터 폴더: `C:\Users\lsed\merlion\data` (`FileManager().data_directory`)
- `black --check`에서 `merlion/dashboard/models/anomaly.py`가 걸린다. 이것도 원래 있던 문제다.
