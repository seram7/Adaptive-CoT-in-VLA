from huggingface_hub import snapshot_download

download_path = snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="/SAN/vision/EastRobotictsVLA/dataset/modified_libero_rlds"
)
print("Files downloaded to:", download_path)