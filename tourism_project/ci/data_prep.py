import os
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME") or os.environ.get("GITHUB_REPOSITORY_OWNER")
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
df = load_dataset(DATASET_REPO, name="raw", split="train", token=HF_TOKEN).to_pandas()
clean = df.drop(columns=[c for c in ["Unnamed: 0","CustomerID"] if c in df.columns])
train_df, test_df = train_test_split(clean, test_size=0.2, random_state=42, stratify=clean["ProdTaken"])
Dataset.from_pandas(train_df, preserve_index=False).push_to_hub(DATASET_REPO, config_name="cleaned", split="train", token=HF_TOKEN)
Dataset.from_pandas(test_df,  preserve_index=False).push_to_hub(DATASET_REPO,  config_name="cleaned", split="test",  token=HF_TOKEN)
print("Uploaded CLEANED train/test to:", DATASET_REPO)
