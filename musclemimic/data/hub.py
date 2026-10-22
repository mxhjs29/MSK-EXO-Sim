"""Optional, explicit dataset download; importing the simulator never downloads."""
def download_dataset(spec):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=spec.repo_id, repo_type="dataset", revision=spec.revision,
                             allow_patterns=list(spec.allow_patterns), local_dir=str(spec.local_dir))
