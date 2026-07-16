# Merlion 코드베이스 분석 (analyze-codebase)

분석일: 2026-07-16
대상: `D:\OpenSpec_CICD\Merlion`

## 1. 의존성과 스택 확인

`setup.py:48-83` 기준:

- **언어**: Python `>=3.7.0`
- **패키지명/버전**: `salesforce-merlion` 2.0.2, BSD-3-Clause 라이선스
- **핵심 의존성** (`install_requires`): `numpy`(`<2.0`, 보안 이유로 `>=1.21` 명시), `pandas`, `scikit-learn`, `scipy`, `statsmodels`, `lightgbm`, `prophet`, `matplotlib`/`plotly`(시각화), **`dill`**(모델 직렬화), `py4j` + JAR 3종(`resources/*.jar`)로 Java(RandomCutForest) 연동, `GitPython`
- **선택적 스택** (`extras_require`):
  - `dashboard` → `dash[diskcache]>=2.4`, `dash_bootstrap_components`, `diskcache` — 웹 GUI 스택은 이 extra에만 존재
  - `deep-learning` → `torch`, `einops`
  - `spark` → `pyspark[sql]>=3`
- **DB/큐/캐시**: SQL/NoSQL DB 의존성 없음. 유일한 "캐시"는 대시보드 롱콜백용 `diskcache`(파일 기반) — 실제 DB는 어디에도 없음
- **테스트**: `pytest.ini`로 설정, live logging 활성화. `tests/`가 `merlion` 서브패키지별로 미러링(`anomaly`, `change_point`, `evaluate`, `forecast`, `transform`, `spark`)
- **서브패키지**: `ts_datasets/setup.py`가 별도 패키지로 존재, editable(`-e`) 설치 필요
- **빌드/포맷**: pre-commit(`black --line-length 120`, `licenseheaders`)으로 커밋 시 강제

## 1-1. 웹서버 백엔드/프론트엔드 구현 프레임워크

이 저장소에는 별도의 백엔드/프론트엔드가 분리 구현되어 있지 않음. 유일한 웹 UI인 `merlion/dashboard/`는 **Dash** 하나로 백엔드·프론트엔드를 모두 처리함.

**백엔드** (`dashboard/server.py:1-41`)
- **Dash** (`dash.Dash(...)`, line 25) — 내부적으로 **Flask**를 WSGI 서버로 사용 (`server = app.server`, line 41 — 실제 Flask 인스턴스). 즉 백엔드 프레임워크는 Flask(Dash가 감싼 형태).
- **`dash_bootstrap_components`** (line 8) — Bootstrap 테마 적용 (`dbc.themes.BOOTSTRAP`, line 28).
- `setup.py`의 `dashboard` extra: `dash[diskcache]>=2.4`, `dash_bootstrap_components>=1.0`, `diskcache` — FastAPI/Django 등 별도 REST 프레임워크는 사용하지 않음.

**프론트엔드**
- Dash가 React 컴포넌트(`dash.dcc`, `dash.html`)를 Python API로 감싸 서버사이드에서 렌더링 — 별도의 JS 프레임워크(React/Vue 등)를 직접 작성하지 않고, Dash가 React 런타임을 내부적으로 자동 번들링함.
- 사용자 정의 프론트 자산은 `dashboard/assets/`(CSS: `base.css`, `merlion.css`, `modal.css`, `styles.css`, JS: `resizing.js`)뿐 — 커스텀 프레임워크 없이 순수 CSS/바닐라 JS 몇 개만 존재.
- 콜백 상태 관리는 `dcc.Store`(`server.py:36-38`)로 처리 — Redux 등 별도 상태관리 라이브러리 없음.

**요약**: 백엔드=Flask(Dash 내장), 프론트엔드=React(Dash 내장) + Bootstrap CSS. 독립된 백엔드 API 서버나 SPA 프론트엔드 코드베이스는 존재하지 않으며, 전부 Dash 단일 프레임워크로 통합되어 있음.

## 1-2. 실행 검증 및 dash 버전 제약사항 (실제 테스트 결과)

`setup.py:36`은 `dashboard` extra에 `dash[diskcache]>=2.4`로 **하한만** 지정하지만, 실제로 `scripts/test_dashboard.py`로 스모크 테스트(`python -m merlion.dashboard` 기동 후 `GET /` 확인)를 돌려본 결과 **상한이 반드시 필요**함이 확인됨:

- `dash 2.18.2` — 정상 동작 (`GET http://127.0.0.1:8050/ -> 200`)
- `dash 3.0.0` 이상 — `merlion/dashboard/utils/file_manager.py`의 `from dash.long_callback import DiskcacheLongCallbackManager`가 `ModuleNotFoundError`로 실패. dash 3.0부터 `dash.long_callback` 모듈이 `dash.background_callback`으로 개명/제거됨.
- `dash 4.4.0`(현재 PyPI 최신) — 동일하게 실패. `pip install -e .[dashboard]`를 그대로 실행하면 최신 dash가 설치되어 **대시보드가 아예 기동되지 않는 실제 버그**.

**실제 필요한 제약**: `dash[diskcache]>=2.4,<3.0` (현재 `setup.py`에는 상한이 없어 부정확함)

**추가로 누락된 의존성**: `dash[diskcache]` extra를 선언해도 `DiskcacheLongCallbackManager`가 내부적으로 요구하는 `psutil`이 함께 설치되지 않아 `ImportError` 발생 — `psutil`을 명시적 의존성으로 추가해야 함.

**검증 환경**: Python 3.14 (numpy<2.0 사전 빌드 휠 없음 → 컴파일러 필요, 우회하여 numpy 2.x/pandas 등 관련 패키지 최신 호환 버전으로 대체 설치 후 검증). `dash==2.18.2` + `psutil` + `multiprocess` 조합으로 스모크 테스트 통과 확인.

## 1-3. Python 버전 재검증 (3.14 → 3.12) 및 실제 코드 버그 수정

Python 3.14 환경에서 발견한 문제들이 Python 버전 자체의 한계인지 merlion 코드의 실제 버그인지 분리하기 위해, **Python 3.12.10**을 별도 venv(`.venv312`)에 설치해 동일 항목을 재검증함.

**재검증 결과 매트릭스**:

| 문제 | Python 3.14 | Python 3.12 |
|---|---|---|
| numpy 빌드 실패(컴파일러 필요) | 실패 | **`setup.py` 선언 그대로(`numpy<2.0`, `lightgbm`, `prophet`, `cython` 포함) 컴파일러 없이 정상 설치** — numpy 문제는 3.14 전용, merlion 코드 문제 아님 |
| dash `>=2.4` 상한 없음 → 최신 dash 설치 시 `ModuleNotFoundError` | 재현 | **동일하게 재현** — Python 버전과 무관한 dash 자체의 버전 문제(1-2 참고) |
| Enum + `functools.partial` (`AggregationPolicy`) | 재현 (`__members__`가 빈 딕셔너리) | **재현 안 됨** — `AnomalyModel`의 실제 학습(`IsolationForest`)이 정상 통과 |

**Enum 버그 최소 재현** (Python 3.14 전용, merlion 코드와 무관하게 인터프리터 자체에서 재현):
```python
from enum import Enum
from functools import partial
class Foo(Enum):
    A = partial(lambda x: x)
print(Foo.__members__)  # Python 3.14: {} / Python 3.12: {'A': <Foo.A>}
```
`merlion/utils/resample.py::AggregationPolicy`가 이 패턴을 사용하므로 Python 3.14에서만 `TimeSeries.align()`이 깨짐 — **merlion 코드 버그가 아니라 Python 3.14 자체의 Enum 구현 변경**.

### 새로 발견되고 수정된 실제 버그: pandas `[-1]` 위치 인덱싱 (KeyError: -1)

Enum 버그가 3.12에서 사라지자, 그 아래 가려져 있던 **별개의 실제 merlion 버그**가 드러남: `ForecastModel.train()`(Arima)이 `pandas.KeyError: -1`로 실패.

**근본 원인**: `DatetimeIndex`를 가진 pandas `Series`에 대해 `series[-1]`(대괄호, 라벨 조회)로 "마지막 값"에 접근하려 시도. 구버전 pandas는 비정수 인덱스에서 정수 키 조회 시 위치 기반으로 암묵 폴백했으나, 이 환경의 **pandas 3.0.3**(`setup.py`는 `pandas>=1.1.0`만 지정, 상한 없음)에서는 이 폴백이 제거되어 `-1`을 존재하지 않는 라벨로 취급 → `KeyError`.

**발견 및 수정 위치** (동일 버그 패턴, 3곳):
- `merlion/models/forecast/sarima.py:112` — `self._last_val = train_data[-1]` → `train_data.iloc[-1]`
- `merlion/models/forecast/sarima.py:131` — `last_val = val_prev[-1]` → `val_prev.iloc[-1]`
- `merlion/models/forecast/ets.py:145` — `self._last_val = train_data[-1]` → `train_data.iloc[-1]`
- 참고: 같은 파일 `ets.py:165`에는 이미 올바른 패턴(`val_prev.iloc[-1]`)이 존재 — 코드베이스 내에서도 일관되지 않게 혼재했음.

**수정 검증**: `tests/dashboard/`의 `ForecastModel` 학습 테스트가 수정 전 `KeyError: -1`로 실패 → 수정 후 Python 3.12에서 통과. (수정 과정에서 테스트 fixture 자체의 결함도 발견: `forecast_train_test_df`가 train/test 구간을 독립적으로 같은 시작일부터 생성해 시간 범위가 어긋났던 문제 — train 구간에 바로 이어지는 연속 구간으로 fixture를 수정함. `merlion/models/forecast/base.py:152-157`이 test `time_stamps`가 `[train_end, train_end+max_forecast_steps]` 범위 내에 있는지 assert하므로, 실제 예측 시나리오처럼 test 구간이 train 구간 직후로 이어져야 함.)

**최종 검증 결과**: Python 3.12 + `numpy<2.0`(선언 그대로) + `dash==2.18.2` 조합에서 `tests/dashboard/` 10개 테스트 전부 통과(`IsolationForest` 이상탐지, `Arima` 예측 모두 학습→저장→로드 전체 흐름 정상). Python 3.14는 6 passed + 4 xfailed(Enum 버그로 인한 조건부 xfail, `sys.version_info >= (3, 14)`).

### 종합: 실행 환경 제약사항

| 제약 | `setup.py` 선언 | 실제 필요 | 근거 |
|---|---|---|---|
| Python | `>=3.7.0` (상한 없음) | **`<3.14`** (정확한 하한 미확인 — 3.12는 안전 확인, 3.13은 미검증) | Enum+`functools.partial` 버그 |
| numpy | `>=1.21,<2.0` | 선언대로 유지 가능 (3.14에서만 컴파일러 문제) | Python 3.12에서 정상 빌드 확인 |
| dash | `>=2.4` (상한 없음) | **`>=2.4,<3.0`** | `dash.long_callback` 모듈 제거 (dash 3.0~) |
| psutil | 미선언 | `dash[diskcache]` 사용 시 명시적으로 필요 | `DiskcacheLongCallbackManager` 런타임 요구사항 |
| pandas | `>=1.1.0` (상한 없음) | 상한 검토 필요 — 최소 `sarima.py`/`ets.py`의 `.iloc[-1]` 수정으로 pandas 3.0.3에서 해결됨(이번 세션에서 수정 완료) | `series[-1]` 위치 인덱싱 폴백 제거 |

**권장 조합**: `Python 3.12 + numpy<2.0 + dash>=2.4,<3.0` — 이번 세션에서 실제로 end-to-end 검증 완료.

## 2. 폴더 구조와 계층 파악

**최상위**: `merlion/`(라이브러리), `data/`, `docs/`, `examples/`, `tests/`, `spark_apps/`, `ts_datasets/`, `conf/`, `docker/`, `k8s-spec/`

**`merlion/` 내부**: `dashboard/`, `evaluate/`, `models/`, `post_process/`, `spark/`, `transform/`, `utils/`, `plot.py`

- `merlion/models/` — 핵심 알고리즘 라이브러리(탐지기/예측기 클래스). 웹 코드 아님.
- `merlion/spark/` — `dataset.py`, `pandas_udf.py` 뿐, Spark 배치 처리용. HTTP와 무관.

**`merlion/dashboard/`**가 routes/controllers/services/db와 가장 유사한 4계층 구조를 가짐(Dash 기반이라 정확히 일치하진 않음):

| 요청한 계층 | 대응하는 것 | 위치 |
|---|---|---|
| routes | Dash 페이지 레이아웃 | `pages/` (`anomaly.py`, `data.py`, `forecast.py`) |
| controllers | Dash `@callback` 함수 | `callbacks/` (`anomaly.py`, `data.py`, `forecast.py`) |
| services | 도메인 로직 래퍼 클래스 | `models/` (`AnomalyModel`, `ForecastModel`, `DataAnalyzer`) — ORM 아님 |
| db | 파일시스템 영속화 | `utils/file_manager.py` (`FileManager`) — DB 없음, 전부 로컬 디스크 |

핵심 라이브러리(`merlion/models`, `merlion/transform` 등)에는 이런 계층 구조가 없고, 대신 알고리즘 모듈 단위(Config/Model/Transform 베이스 클래스)로 구성됨.

## 3. 핵심 데이터 모델 지목 (도메인 언어)

- **`ModelBase`/`Config`** (`merlion/models/base.py`) — 모든 모델이 구현하는 `save()`/`load()` 영속화 계약 (`ModelBase.save` at line 368, `ModelBase.load` at line 405)
- **`ModelFactory`** (`merlion/models/factory.py`) — 알고리즘 이름 문자열 → 실제 탐지기/예측기 클래스 매핑
- **`TimeSeries`** (`merlion/utils/time_series.py`) — load→train→test 전 구간에서 전달되는 핵심 데이터 구조
- **`AnomalyModel`** (`dashboard/models/anomaly.py`) — 이상탐지 알고리즘의 train/test/plot 로직 래퍼
- **`ForecastModel`** (`dashboard/models/forecast.py`) — 예측 알고리즘의 동일 패턴 래퍼
- **`DataAnalyzer`** (`dashboard/models/data.py`) — 데이터 페이지용 통계 계산
- **`ModelMixin`/`DataMixin`** (`dashboard/models/utils.py`) — 공통 메서드: `get_parameter_info`, `parse_parameters`, `save_model`, `load_model`, `load_data`
- **`FileManager`** (`dashboard/utils/file_manager.py`) — `~/merlion` 하위 `data_folder`/`model_folder`/`cache_folder`를 관리하는 프로세스 전역 싱글턴
- **`Threshold`/`AggregateAlarms`** (`merlion/post_process/threshold.py`) — 이상탐지 점수 후처리 설정 객체

## 4. 주요 엔드포인트와 상태 코드 정리

Dash 앱이므로 실제 HTTP 라우트/상태코드는 없음. `dashboard/server.py`는 `dash.Dash` 앱 하나만 생성하고(`@server.route`, `add_url_rule` 등록 없음), 모델 다운로드도 `dcc.send_file`을 콜백 `Output`으로 흘려보내는 방식일 뿐 별도 엔드포인트가 아님.

**성공/실패 신호 방식**:
- `dash.exceptions.PreventUpdate`는 전혀 사용되지 않음 — 대신 `ctx.triggered_id`로 분기하고, 조건 미충족 시 초기값(빈 리스트/None)을 반환하는 "no-op" 패턴
- 조회성 콜백(`update_select_file_dropdown` 등, `data.py:37`, `anomaly.py:29`)은 try/except가 아예 없음 — 예외 시 콜백이 그대로 실패하고 브라우저 콘솔에 raw 에러 노출
- 학습/로드 등 핵심 콜백(`data.py:88`, `anomaly.py:287`, `forecast.py:235`)은 `try/except Exception`으로 감싸 `traceback.format_exc()`를 `*-exception-modal` Output에 렌더링하고 `logger.error`로 기록

## 5. 한 요청의 흐름 끝까지 추적 (업로드 → 학습 → 저장)

1. **업로드 (POST 유사)**: `dashboard/callbacks/data.py::upload_file()` — Dash `Input("upload-data", "contents")` 콜백 → `FileManager.save_file()` 호출
2. **원본 저장**: `dashboard/utils/file_manager.py::FileManager.save_file()` — base64 디코딩 후 CSV를 `~/merlion/data/`에 저장
3. **학습 버튼 클릭**: `dashboard/callbacks/anomaly.py::click_train_test()`
   - `DataMixin.load_data()`로 CSV 로드
   - `ModelMixin.get_parameter_info`/`parse_parameters`로 알고리즘·파라미터 확정
   - `AnomalyModel.train()` (`dashboard/models/anomaly.py`) → `ModelFactory.get_model_class(algorithm)`로 실제 탐지기 인스턴스화 후 `.train(train_data=...)` 호출
4. **학습된 모델 저장**: `ModelMixin.save_model()` → `model.save(dir)` 호출
5. **실제 디스크 쓰기**: `merlion/models/base.py::ModelBase.save()` (368번째 줄) — 모델 상태/config를 디렉터리에 직렬화 저장

로드 경로도 대칭적으로 `AnomalyModel.load_model` → `ModelFactory` → `ModelBase.load()` (405번째 줄)로 이어짐. DB는 어디에도 없고, 모든 영속화는 `FileManager`가 관리하는 로컬 파일시스템 경로를 통해 이뤄짐.

## 5-1. Dashboard ↔ Merlion 코어 모듈 인터페이스 분석

`merlion/dashboard/`는 core 라이브러리와 세 개의 결합 지점(interface)을 통해서만 연결됨. 이 세 인터페이스만 지키면 core에 새 모델을 추가해도 dashboard 코드 변경이 불필요.

**연결 구조**:
```
dashboard/callbacks/anomaly.py, forecast.py
  → dashboard/models/anomaly.py (AnomalyModel), forecast.py (ForecastModel)
      → merlion.models.factory.ModelFactory        (algorithm 문자열 → 모델 클래스)
      → merlion.utils.time_series.TimeSeries        (DataFrame ↔ 내부 시계열 구조)
      → merlion.models.*.<Model>Config / <Model>    (실제 탐지기/예측기)
      → merlion.post_process.threshold.*            (importlib 동적 로드, 임계값 후처리)
      → merlion.evaluate.anomaly.TSADMetric /
        merlion.evaluate.forecast.ForecastEvaluator  (성능 지표 계산)
      → merlion.plot.MTSFigure / plot_anoms_plotly   (시각화)
```

**인터페이스별 소스 위치**:

- **`ModelFactory`** — `merlion/models/factory.py:74` `class ModelFactory`, `get_model_class(cls, name: str) -> Type[ModelBase]` (line 76). 알고리즘 이름 문자열을 모델 클래스로 매핑하는 레지스트리. 대시보드 호출부: `dashboard/models/anomaly.py:104`, `dashboard/models/forecast.py:12`.
- **`TimeSeries`** — `merlion/utils/time_series.py:329` `class TimeSeries`. 변환 진입점은 `TimeSeries.from_pd(cls, df, check_times=True, drop_nan=True, freq="1h")` (line 759) — pandas DataFrame/Series/ndarray → `TimeSeries` 변환. (참고: 단변량 전용 `UnivariateTimeSeries.from_pd`는 line 286에 별도 존재.) 대시보드 호출부: `dashboard/models/anomaly.py:106-110`.
- **`ForecastEvaluator`** — `merlion/evaluate/forecast.py`: `ForecastMetric(Enum)` (line 240, sMAPE/MSIS 등 지표), `ForecastEvaluatorConfig` (line 304), `ForecastEvaluator(EvaluatorBase)` (line 347).
- **`ModelBase`** (위 세 인터페이스가 실제로 다루는 타입) — `merlion/models/base.py:154` `class ModelBase`, `train(self, train_data: TimeSeries, train_config=None)` (line 333). `ModelFactory.get_model_class()`가 반환하는 타입이 곧 이 클래스의 서브클래스.

**대시보드 학습 흐름 예시** (`dashboard/models/anomaly.py:94-133`, `AnomalyModel.train`):

1. `ModelFactory.get_model_class(algorithm)` (line 104) — 문자열 → 클래스
2. `model_class(model_class.config_class(**params))` (line 105) — UI 파라미터를 core의 `Config` 클래스에 그대로 주입 (대시보드는 개별 모델 파라미터 스키마를 모름, core가 `config_class`로 스스로 노출)
3. `importlib.import_module("merlion.post_process.threshold")` + `getattr(module, thres_class)` (line 100-102) — 임계값 클래스는 `ModelFactory`와 별개로 자체 리플렉션 사용 (동일 목적에 두 가지 동적 로딩 패턴이 혼재)
4. `TimeSeries.from_pd(train_df[columns])` (line 106) — GUI 표현(DataFrame)과 core 표현(TimeSeries)의 경계
5. `model.train(train_data=train_ts)` (line 115), `model.get_anomaly_label(test_ts)` (line 124) — 실제 알고리즘 로직은 전부 core에 위치, dashboard는 호출만 함
6. `AnomalyModel._compute_metrics` → `merlion.evaluate.anomaly.TSADMetric` — core 평가 파이프라인 재사용
7. `merlion.plot.MTSFigure`/`plot_anoms_plotly` — core plotting 유틸 재사용

`forecast.py`도 동일 패턴(`ModelFactory`, `ForecastEvaluator`, `TimeSeries`)으로 결합됨.

**관찰**: 결합 지점이 `ModelFactory`(모델 종류) / `TimeSeries`(데이터 형식) / `Config`·`ModelBase`(파라미터·영속화 계약) 세 곳으로 명확히 제한되어 있어 확장성은 좋으나, 임계값(threshold) 클래스 로딩만 `ModelFactory` 레지스트리를 쓰지 않고 별도의 `importlib` 리플렉션을 사용하는 비일관성이 존재함.

## 6. 숨은 규약과 약점 발견

1. **역직렬화 위험 (dill/pickle 기반 모델 로드)** — `merlion/models/base.py::ModelBase.load()`가 `dill`로 임의 모델 디렉토리를 역직렬화. 서명 검증·화이트리스트 없음 → 신뢰할 수 없는 모델 파일 로드 시 임의 코드 실행 가능.
2. **경로 탈출 (path traversal)** — `dashboard/utils/file_manager.py:39-42 FileManager.save_file`이 업로드 파일명을 sanitize 없이 `os.path.join`에 사용. `../`가 포함된 파일명으로 `data_folder` 밖에 쓰기 가능.
3. **에러 응답의 정보 노출** — `data.py:88`, `anomaly.py:287`, `forecast.py:235`의 `except Exception` 블록이 `traceback.format_exc()`를 그대로 모달에 표시 → 서버 내부 경로/스택 노출.
4. **전역 싱글턴 상태 공유 (다중 사용자 격리 없음)** — `FileManager`(`file_manager.py:15-19`)가 프로세스 전체에서 하나뿐이라 모든 대시보드 세션이 같은 `~/merlion/data`, `~/merlion/models`를 공유 → 세션 간 파일 노출·덮어쓰기 충돌.
5. **다운로드 zip 레이스 컨디션** — `get_model_download_path`(`file_manager.py:52-59`)가 동일 경로에 매번 덮어쓰기 압축 → 동시 다운로드 시 zip 파일 손상 가능.
6. **조회성 콜백의 무방비 예외 전파** — `update_select_file_dropdown` 등(`data.py:37`, `anomaly.py:29`)은 try/except가 전혀 없어 예외 시 브라우저 콘솔에 raw 에러 노출(정보 노출은 적지만 UX 저하).
7. **초기화 순서 암묵 의존** — `diskcache.Cache` 생성(`file_manager.py:37`)이 디렉토리 존재 여부를 확인하지 않아, import 순서에 조용히 의존.
8. **[수정 완료] pandas 버전 비호환 — `series[-1]` 위치 인덱싱** — `sarima.py:112,131`, `ets.py:145`가 `DatetimeIndex` 기반 `Series`에 `[-1]`(라벨 조회)로 접근해 pandas 3.0.3에서 `KeyError: -1` 발생. `.iloc[-1]`로 수정하여 해결(1-3 참고). `setup.py`의 `pandas>=1.1.0`에 상한이 없어 향후 pandas 신버전에서 유사 문제가 재발할 수 있음 — 코드베이스 전체에 동일 패턴(`[-1]`, `[0]` 등 라벨/위치 혼용)이 더 있는지 전수 점검 필요.
9. **`FileManager(directory=...)` 생성자 인자가 실제로 동작하지 않음** — `SingletonClass.__new__(cls)`가 인자를 받지 않아 `FileManager(custom_dir)` 호출 시 `TypeError`. 프로덕션 코드는 항상 `FileManager()`(무인자)로만 호출해 지금까지 발견되지 않은 dormant 버그. (`tests/dashboard/conftest.py`의 `file_manager` fixture에서 `object.__new__` 우회로 확인 및 대응.)

## 요약

Merlion 핵심 라이브러리는 웹 계층 없는 순수 Python 시계열 라이브러리이며, DB도 없다. `merlion/dashboard/`만이 routes/controllers/services/db에 대응하는 유사 구조(pages→callbacks→models→file_manager)를 가지고 있고, 모든 영속화는 로컬 파일시스템 기반이다. 이 대시보드는 단일 사용자/로컬 실행을 전제로 설계된 것으로 보이며, 다중 사용자 환경에 그대로 노출하면 파일 충돌·정보 노출·역직렬화 기반 RCE 위험이 실질적인 문제가 된다.

**실행 환경 검증 이력**: Python 3.14에서 numpy 빌드 실패, dash 버전 문제, Enum 호환성 문제(3.14 전용, 실질적으로 모든 모델 학습을 막음)를 발견. Python 3.12로 재검증한 결과 numpy/Enum 문제는 해소되었으나, 그 아래 가려져 있던 별개의 실제 버그(pandas `[-1]` 위치 인덱싱, `sarima.py`/`ets.py`)를 새로 발견해 `.iloc[-1]`로 수정 완료. 최종적으로 `Python 3.12 + numpy<2.0 + dash>=2.4,<3.0` 조합에서 이상탐지·예측 모델의 학습→저장→로드 전체 흐름을 `tests/dashboard/`(10개 테스트)로 end-to-end 검증함.
