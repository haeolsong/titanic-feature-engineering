"""
과제3: 특성 공학(Feature Engineering) 파이프라인 모듈

- add_features()   : 파생 변수 생성 (FamilySize, IsAlone, FarePerPerson, Title, HasCabin)
- build_pipeline() : 결측치 처리 / 인코딩 / 스케일링 / 변수 선택 / 모델을 하나의 sklearn Pipeline으로 조립
- get_model()      : 비교 모델 4종 (Logistic Regression, Random Forest, XGBoost, LightGBM)

모든 전처리는 Pipeline 내부에서 fit 되므로 CV fold마다 train fold로만 학습된다 (데이터 누수 방지).
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, StandardScaler,
                                   MinMaxScaler, RobustScaler, FunctionTransformer, TargetEncoder)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
TARGET = "Survived"

# ---------------------------------------------------------------- 컬럼 정의
BASE_NUM_COLS = ["Pclass", "SibSp", "Parch", "Fare"]          # 결측 없는 원본 수치형 (Base 실험용)
RAW_NUM_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
RAW_CAT_COLS = ["Sex", "Embarked"]
FE_NUM_COLS = ["FamilySize", "IsAlone", "FarePerPerson", "HasCabin"]
FE_CAT_COLS = ["Title"]
AGE_BINS = [0, 12, 18, 35, 60, 200]                            # AgeGroup: Child/Teen/YoungAdult/Adult/Senior


# ---------------------------------------------------------------- 파생 변수
def extract_title(name: pd.Series) -> pd.Series:
    title = name.str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """행 단위로만 계산하는 파생 변수 (다른 행의 통계를 쓰지 않으므로 split 전에 만들어도 누수 없음)."""
    out = df.copy()
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
    out["Title"] = extract_title(out["Name"])
    out["HasCabin"] = out["Cabin"].notna().astype(int)        # 결측 자체를 정보로 사용
    return out


def _age_to_group(X):
    """결측 대체가 끝난 Age를 구간 번호(0~4)로 변환. 대체 이후에 실행되도록 파이프라인 안에 둔다."""
    X = np.asarray(X, dtype=float)
    return np.digitize(X, AGE_BINS[1:-1])


def _agegroup_names(_, __):
    return np.array(["AgeGroup"])


# ---------------------------------------------------------------- 파이프라인 조립
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def _encoder(kind):
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "label":   # 입력 변수용 Label Encoding = OrdinalEncoder (LabelEncoder는 타겟 전용)
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if kind == "target":  # 범주별 생존율(평활화)로 치환. fit 시 내부 5-fold cross-fitting으로 타겟 누수 방지
        return TargetEncoder(target_type="binary", random_state=RANDOM_STATE)
    raise ValueError(kind)


def get_model(name):
    if name == "LogReg":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, eval_metric="logloss",
                             random_state=RANDOM_STATE, n_jobs=1)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    raise ValueError(name)


MODEL_NAMES = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]


RAW_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]   # 자유 텍스트(Name, Ticket, Cabin) 제외 원본


def to_raw(df):
    """Base-Raw 입력: 원본 7개 컬럼을 결측·문자열 그대로 둔다 (문자열은 pandas category 타입 표시만)."""
    d = df[RAW_COLS].copy()
    for c in ["Sex", "Embarked"]:
        d[c] = d[c].astype("category")
    return d


def get_native_model(name):
    """Base-Raw용: 전처리 없이 원본(결측·범주형 포함)을 그대로 학습. XGBoost/LightGBM만 자체 처리 가능."""
    m = get_model(name)
    if name == "XGBoost":
        m.set_params(enable_categorical=True, tree_method="hist")
    return m


def build_base_pipeline(model_name):
    """Base: 전처리 없음. 결측이 없는 원본 수치형 4개 컬럼만 그대로 사용 (인코딩·스케일링·파생변수 없음)."""
    pre = ColumnTransformer([("num", "passthrough", BASE_NUM_COLS)])
    return Pipeline([("pre", pre), ("clf", get_model(model_name))])


def build_pipeline(model_name, impute="mean", encode="onehot", scale="standard",
                   fs_k=None, use_fe=True):
    """
    impute : 'mean' | 'median' | 'most_frequent'  (수치형 대체 전략. 범주형은 항상 최빈값)
             'none' 은 Drop NA 실험용 (사전에 결측 행을 제거했을 때)
    encode : 'onehot' | 'label' | 'target'
    scale  : 'standard' | 'minmax' | 'robust' | None
    fs_k   : None 이면 변수 선택 없음, 정수면 RandomForest 중요도 상위 k개만 사용
    use_fe : 파생 변수(FamilySize, IsAlone, FarePerPerson, HasCabin, Title, AgeGroup) 사용 여부
    """
    num_cols = RAW_NUM_COLS + (FE_NUM_COLS if use_fe else [])
    cat_cols = RAW_CAT_COLS + (FE_CAT_COLS if use_fe else [])
    num_strategy = "median" if impute == "none" else impute     # none: 결측이 없으므로 대체가 실제로 일어나지 않음

    num_steps = [("imp", SimpleImputer(strategy=num_strategy))]
    if scale:
        num_steps.append(("sc", SCALERS[scale]()))
    cat_steps = [("imp", SimpleImputer(strategy="most_frequent")), ("enc", _encoder(encode))]

    transformers = [("num", Pipeline(num_steps), num_cols),
                    ("cat", Pipeline(cat_steps), cat_cols)]
    if use_fe:
        # AgeGroup: Age 결측 대체 -> 구간화 -> 인코딩 (대체 이후 구간화해야 하므로 별도 가지로 구성)
        age_steps = [("imp", SimpleImputer(strategy=num_strategy)),
                     ("bin", FunctionTransformer(_age_to_group, feature_names_out=_agegroup_names))]
        if encode == "onehot":
            age_steps.append(("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                   categories=[list(range(len(AGE_BINS) - 1))])))
        transformers.append(("agegrp", Pipeline(age_steps), ["Age"]))

    steps = [("pre", ColumnTransformer(transformers))]
    if fs_k:
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            threshold=-np.inf, max_features=fs_k)
        steps.append(("fs", selector))
    steps.append(("clf", get_model(model_name)))
    return Pipeline(steps)
