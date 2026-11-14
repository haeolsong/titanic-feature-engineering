# [과제3] 특성 공학(Feature Engineering) 파이프라인 구현 및 성능 비교 실험

빅데이터분석 | 동아대학교 컴퓨터공학과 송하얼 (2337310)

Titanic 데이터로 결측치 처리 · 범주형 인코딩 · 스케일링 · 파생 변수 · 변수 선택 전략을 바꿔가며
4개 분류 모델(Logistic Regression, Random Forest, XGBoost, LightGBM)의 성능을 비교한다.
모든 전처리는 `sklearn.Pipeline` + `ColumnTransformer` 안에서 수행되어 CV fold마다 train fold로만 학습된다.

## 실행 방법 (재현)
```bash
pip install -r requirements.txt        # Python 3.10+ (실험 환경: Python 3.13)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_experiments.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_automl.ipynb
```
- 모든 난수는 `RANDOM_STATE = 42`로 고정 (분할, CV, 모델, permutation importance)
- 2코어 CPU 기준 실행 시간: 02 노트북 약 5분 (GridSearchCV 포함), 03 노트북 약 1분

## 데이터
- `data/train.csv` : Kaggle *Titanic - Machine Learning from Disaster* 의 `train.csv` (891행 × 12열)
  - 원본: https://www.kaggle.com/competitions/titanic/data
  - Kaggle `test.csv`는 정답이 없어 사용하지 않고, `train.csv`를 Stratified 80:20(712/179)으로 분할
  - 무결성 확인: SHA-256 `4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7`
    (Kaggle Titanic train.csv를 그대로 보관한 공개 저장소 agconti/kaggle-titanic의 `data/train.csv`와 바이트 단위로 동일)
  - 확인 명령: `shasum -a 256 data/train.csv` (macOS) / `sha256sum data/train.csv` (Linux)

## 폴더 구조
```
├── data/train.csv
├── src/fe_pipeline.py          # 파생 변수 생성 + 파이프라인 조립 함수 + 모델 정의
├── notebooks/
│   ├── 01_eda.ipynb            # STEP 01~02: 데이터 구조, 결측치, 이상치, 분포, 상관관계, 타겟 분포
│   ├── 02_experiments.ipynb    # STEP 03~05: 특성 공학 비교, 변수 선택, 모델 비교, GridSearchCV, SHAP
│   └── 03_automl.ipynb         # 가산점: AutoML(FLAML) vs 수동 파이프라인 비교
├── results/
│   ├── main_comparison.csv     # 필수 실험표 (Base / Exp-1 / Exp-2 / Exp-3 × 4 모델)
│   ├── all_results.csv         # 모든 실험(ablation 포함)의 CV 지표 5종 + 과적합 gap
│   ├── automl_comparison.csv   # AutoML(원본 입력 / 파생 변수 입력) vs 수동 파이프라인 · GridSearchCV
│   ├── gridsearch_results.csv, fs_k_curve.csv, rf_feature_importance.csv, shap_importance.csv
│   └── figures/                # 보고서에 사용한 모든 그림
└── report/report.pdf           # 제출 보고서
```

## 실험 구성
| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection |
|---|---|---|---|---|
| Base-Raw | 없음 | 없음 | 없음 | 없음 (원본 7개 컬럼을 결측·문자열 그대로 입력 — XGBoost/LightGBM만 학습 가능) |
| Base-Num | 없음 | 없음 | 없음 | 없음 (4개 모델 비교용: 결측 없는 원본 수치형 4개 컬럼만 사용) |
| Exp-1 | Mean | One-Hot | Standard | X |
| Exp-2 | Median | Label(Ordinal) | MinMax | O (RF 중요도 상위 10개) |
| Exp-3 | Most Frequent | One-Hot | Robust | O (RF 중요도 상위 10개) |

추가로 한 번에 한 요소만 바꾸는 ablation(결측치 4종, 인코딩 3종(One-Hot/Label/Target), 스케일링 4종, 파생 변수 유무, 변수 선택 유무)을 수행했다.
파생 변수: FamilySize, IsAlone, FarePerPerson, HasCabin, Title, AgeGroup

가산점 항목: Pipeline 객체, GridSearchCV, SHAP, Feature Importance 시각화 고도화(MDI vs Permutation), AutoML(FLAML) 비교 실험 — 5개 모두 수행
