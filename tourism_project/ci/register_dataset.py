import os, pandas as pd
from datasets import Dataset
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME") or os.environ.get("GITHUB_REPOSITORY_OWNER")
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
df = pd.read_csv("tourism_project/data/tourism.csv")
Dataset.from_pandas(df, preserve_index=False).push_to_hub(DATASET_REPO, config_name="raw", split="train", token=HF_TOKEN)
print("Uploaded RAW to:", DATASET_REPO)
