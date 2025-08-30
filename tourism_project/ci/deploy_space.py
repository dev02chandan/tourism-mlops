import os
from huggingface_hub import HfApi, upload_folder
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME") or os.environ.get("GITHUB_REPOSITORY_OWNER")
SPACE_REPO = f"{HF_USERNAME}/tourism-wellness-space"
api = HfApi(); api.create_repo(SPACE_REPO, repo_type="space", private=True, exist_ok=True, space_sdk="docker", token=HF_TOKEN)
upload_folder(folder_path="tourism_project/deployment", repo_id=SPACE_REPO, repo_type="space", token=HF_TOKEN)
print("Updated Space:", SPACE_REPO)
