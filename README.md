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
```
- 모든 난수는 `RANDOM_STATE = 42`로 고정 (분할, CV, 모델, permutation importance)
- 2코어 CPU 기준 02 노트북 실행 시간 약 5분 (GridSearchCV 포함)

## 데이터
- `data/train.csv` : Kaggle *Titanic - Machine Learning from Disaster* 의 `train.csv` (891행 × 12열)
  - 원본: https://www.kaggle.com/competitions/titanic/data
  - Kaggle `test.csv`는 정답이 없어 사용하지 않고, `train.csv`를 Stratified 80:20(712/179)으로 분할

## 폴더 구조
```
├── data/train.csv
├── src/fe_pipeline.py          # 파생 변수 생성 + 파이프라인 조립 함수 + 모델 정의
├── notebooks/
│   ├── 01_eda.ipynb            # STEP 01~02: 데이터 구조, 결측치, 이상치, 분포, 상관관계, 타겟 분포
│   └── 02_experiments.ipynb    # STEP 03~05: 특성 공학 비교, 변수 선택, 모델 비교, GridSearchCV, SHAP
├── results/
│   ├── main_comparison.csv     # 필수 실험표 (Base / Exp-1 / Exp-2 / Exp-3 × 4 모델)
│   ├── all_results.csv         # 모든 실험(ablation 포함)의 CV 지표 5종 + 과적합 gap
│   ├── gridsearch_results.csv, fs_k_curve.csv, rf_feature_importance.csv, shap_importance.csv
│   └── figures/                # 보고서에 사용한 모든 그림
└── report/report.pdf           # 제출 보고서
```

## 실험 구성
| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection |
|---|---|---|---|---|
| Base | 없음 | 없음 | 없음 | 없음 (결측 없는 원본 수치형 4개 컬럼만 사용) |
| Exp-1 | Mean | One-Hot | Standard | X |
| Exp-2 | Median | Label(Ordinal) | MinMax | O (RF 중요도 상위 10개) |
| Exp-3 | Most Frequent | One-Hot | Robust | O (RF 중요도 상위 10개) |

추가로 한 번에 한 요소만 바꾸는 ablation(결측치 4종, 인코딩 2종, 스케일링 4종, 파생 변수 유무, 변수 선택 유무)을 수행했다.
파생 변수: FamilySize, IsAlone, FarePerPerson, HasCabin, Title, AgeGroup
