"""Append the provider's Gait120 files to an existing HF dataset, preserving other data."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFolder, RepoFile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "Gait120_101_to_110"


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(block); git.update(block)
    return sha.hexdigest(), git.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(source, staging):
    rows = []
    for file in sorted(source.rglob("*")):
        if not file.is_file() or file.name == ".DS_Store" or any(x in file.parts for x in [".cache", "__pycache__", ".git"]):
            continue
        if file.suffix in {".pyc", ".lock", ".incomplete", ".metadata"}:
            continue
        name = str(file.relative_to(source))
        target = staging / PREFIX / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try: os.link(file, target)
            except OSError: shutil.copy2(file, target)
        sha, git = hashes(file)
        rows.append({"path": PREFIX + "/" + name, "bytes": file.stat().st_size, "sha256": sha, "git_blob_sha1": git})
    save_json(staging / PREFIX / "file_manifest.json", {"source_definition": "Provider-distributed Gait120 files, no project MSK retargeting.", "files": rows})
    (staging / PREFIX / "README.md").write_text(f'''# Gait120 S101–S110 原始来源数据

此目录补充到 `mxhjs29/musclemimic-msk`，不会替换仓库已有 KIT/CMU/Transitions/BABEL 数据。
保留当前下载的 S101–S110 提供者文件，共 {len(rows):,} 个、{sum(r['bytes'] for r in rows):,} 字节；MOT/TRC/MAT 的原文件内容不变。
“原始”指未经本项目肌骨重映射的提供者数据包，包中可含提供者发布的处理表。
逐文件原始 SHA256 见 `file_manifest.json`。不含本项目重映射、训练输出、checkpoint 或 SMPL/MANO 模型。
此目录是本机已有受试者子集，不声明为完整 Gait120；使用和分发仍遵循来源数据条款。
''')
    return rows


def root_objects(api, repo, revision):
    values = {}
    for entry in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=False):
        if entry.path == PREFIX or entry.path == ".gitattributes":
            continue
        values[entry.path] = entry.tree_id if isinstance(entry, RepoFolder) else entry.blob_id
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",default="mxhjs29/musclemimic-msk")
    p.add_argument("--source",type=Path,default=ROOT/"datasets"/PREFIX)
    p.add_argument("--staging",type=Path,default=ROOT/"outputs/hub_publish_20261013/gait120_upload")
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args();args.staging.mkdir(parents=True,exist_ok=True)
    rows=prepare(args.source,args.staging)
    print(json.dumps({"event":"prepared","files":len(rows),"bytes":sum(r['bytes'] for r in rows)}),flush=True)
    if args.prepare_only:return
    api=HfApi();before=api.dataset_info(args.repo)
    original=root_objects(api,args.repo,before.sha)
    baseline_path=args.staging.parent/"hf_before.json"
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text())
        assert baseline["repo_id"]==args.repo and baseline["root_objects"]==original,"Original repository content changed before resume"
    else:
        save_json(baseline_path,{"repo_id":args.repo,"revision":before.sha,"private":before.private,"root_objects":original})
        baseline=json.loads(baseline_path.read_text())
    api.upload_large_folder(repo_id=args.repo,repo_type="dataset",folder_path=args.staging,
        allow_patterns=[PREFIX+"/**"],num_workers=args.workers,print_report_every=30)
    after=api.dataset_info(args.repo)
    remote={entry.path:entry for entry in api.list_repo_tree(args.repo,path_in_repo=PREFIX,repo_type="dataset",revision=after.sha,recursive=True) if isinstance(entry,RepoFile)}
    for row in rows:
        entry=remote[row['path']]
        assert entry.size==row['bytes'],row['path']
        if entry.lfs is not None:assert entry.lfs.sha256==row['sha256'],row['path']
        else:assert entry.blob_id==row['git_blob_sha1'],row['path']
    assert root_objects(api,args.repo,after.sha)==original,"Existing non-Gait120 repository content changed"
    record={"repo_id":args.repo,"revision":after.sha,"previous_revision":baseline["revision"],"private":after.private,
        "added_prefix":PREFIX,"source_files":len(rows),"source_bytes":sum(r['bytes'] for r in rows),
        "remote_hashes_verified":True,"existing_data_unchanged":True,"completed_at":time.time()}
    save_json(args.staging.parent/"gait120_completed.json",record)
    print(json.dumps({"event":"completed",**record}),flush=True)


if __name__=="__main__":main()
