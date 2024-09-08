import argparse
from huggingface_hub import (
    create_repo,
    get_full_repo_name,
    upload_file,
)

def create_and_upload_to_space(files, hf_token, target_space_name):
    # Create the space repository
    create_repo(name=target_space_name, token=hf_token, repo_type="space", space_sdk="gradio")

    # Get the full repository name
    repo_name = get_full_repo_name(model_id=target_space_name, token=hf_token)

    # Upload the file to the space repository
    for file_path in files:
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id=repo_name,
            repo_type="space",
            token=hf_token,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Hugging Face space and upload a file to it.")

    parser.add_argument("--hf_token", required=True, help="Your Hugging Face token")
    parser.add_argument("--target_space_name", required=True, help="The name of the target space")

    args = parser.parse_args()

    # File to upload
    files = [
        "app.py",
        "requirements.txt",
    ]

    # Call the function with arguments from the command line
    create_and_upload_to_space(
        files=files,
        hf_token=args.hf_token,
        target_space_name=args.target_space_name
    )

    print(f"Files uploaded to space {args.target_space_name} successfully.")
