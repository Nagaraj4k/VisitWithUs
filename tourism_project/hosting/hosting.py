from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# 1. Initialize API with Token
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found. Check your environment variables!")

api = HfApi(token=token)

# 2. MATCHING YOUR SCREENSHOT EXACTLY
# The owner is you (Nagaraj4k) and the Space name is Visit-With-Us
repo_id = "Nagaraj4k/Visit-With-Us" 
repo_type = "space"

# 3. Space Management
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Found existing Space: {repo_id}")
except RepositoryNotFoundError:
    print(f"🚀 Space '{repo_id}' not found. Creating it now...")
    # Matches the 'Docker' choice from your screenshot
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="docker", private=False)

# 4. Upload Files
deployment_folder = os.path.abspath("/content/tourism_project/hosting")

api.upload_folder(
    folder_path=deployment_folder,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="", 
    commit_message="MLOps: Deploying Visit-With-Us Wellness Predictor",
    ignore_patterns=["hosting.py", "__pycache__"] 
)

print(f"\n🎉 SUCCESS! Access your app: https://huggingface.co/spaces/{repo_id}")
