from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Updated to your project naming convention
repo_id = "Nagaraj4k/VisitWithUs-Dataset"
repo_type = "dataset"

# Initialize API client using the Secret Token from your GitHub Actions
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found. Ensure it is set in GitHub Secrets.")

api = HfApi(token=token)

# Step 1: Check if the repository exists on Hugging Face
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repository...")
    # Creating as public so the pipeline can access it easily
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created successfully.")

# Step 2: Upload the processed tourism data
# Updated folder_path to match your project structure
api.upload_folder(
    folder_path="tourism_project/data", 
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Updating Wellness Tourism dataset for MLOps pipeline"
)

print(f"Data successfully registered at: https://huggingface.co/datasets/{repo_id}")
