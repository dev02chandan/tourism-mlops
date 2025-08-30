import os, json, numpy as np, joblib, pandas as pd, pathlib
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, upload_file
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME") or os.environ.get("GITHUB_REPOSITORY_OWNER")
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO = f"{HF_USERNAME}/tourism-wellness-model"

train_df = load_dataset(DATASET_REPO, name="cleaned", split="train", token=HF_TOKEN).to_pandas()
test_df  = load_dataset(DATASET_REPO, name="cleaned", split="test",  token=HF_TOKEN).to_pandas()
X_train, y_train = train_df.drop(columns=["ProdTaken"]), train_df["ProdTaken"].astype(int)
X_test,  y_test  = test_df.drop(columns=["ProdTaken"]),  test_df["ProdTaken"].astype(int)

num = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat = [c for c in X_train.columns if c not in num]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", random_state=42))])
gs = GridSearchCV(pipe, {"clf__n_estimators":[200,300], "clf__max_depth":[None,20], "clf__min_samples_split":[2,5]},
                  cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=-1, refit=True)
gs.fit(X_train, y_train)
best = gs.best_estimator_

proba = best.predict_proba(X_test)[:,1]; pred = (proba>=0.5).astype(int)
metrics = {{
  "model":"random_forest",
  "test_auc": float(roc_auc_score(y_test, proba)),
  "test_accuracy": float(accuracy_score(y_test, pred)),
  "test_precision": float(precision_score(y_test, pred, zero_division=0)),
  "test_recall": float(recall_score(y_test, pred, zero_division=0)),
  "test_f1": float(f1_score(y_test, pred, zero_division=0)),
  "best_params": gs.best_params_
}}

mb = pathlib.Path("tourism_project/model_building"); mb.mkdir(parents=True, exist_ok=True)
joblib.dump(best, mb/"best_model.joblib"); (mb/"metrics.json").write_text(json.dumps(metrics, indent=2))

api = HfApi(); api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True, token=HF_TOKEN)
open("README.md","w").write(f"# Tourism Wellness Model\nTest AUC: {metrics['test_auc']:.4f}\n")
upload_file(path_or_fileobj=str(mb/'best_model.joblib'), path_in_repo='best_model.joblib', repo_id=MODEL_REPO, repo_type='model', token=HF_TOKEN)
upload_file(path_or_fileobj=str(mb/'metrics.json'),      path_in_repo='metrics.json',      repo_id=MODEL_REPO, repo_type='model', token=HF_TOKEN)
print("Registered model at:", MODEL_REPO)
