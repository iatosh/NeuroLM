#!/usr/bin/env python3
"""
NeuroLM Checkpoint Downloader
HuggingFace hubからNeuroLMの全てのチェックポイントをダウンロードします
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

def download_neurolm_checkpoints(local_dir="./checkpoints"):
    """
    NeuroLMの全てのチェックポイントをHuggingFace Hubからダウンロード
    
    Args:
        local_dir (str): ダウンロード先ディレクトリ
    """
    repo_id = "Weibang/NeuroLM"
    
    # ローカルディレクトリを作成
    local_path = Path(local_dir)
    local_path.mkdir(exist_ok=True)
    
    print(f"📁 Creating directory: {local_path.absolute()}")
    
    # HuggingFace APIを初期化
    api = HfApi()
    
    try:
        # リポジトリ内の全てのファイル情報を取得
        print(f"🔍 Fetching file list from {repo_id}...")
        repo_files = api.list_repo_files(repo_id=repo_id)
        
        # .ptファイル（PyTorchチェックポイント）をフィルタリング
        checkpoint_files = [f for f in repo_files if f.endswith('.pt')]
        
        if not checkpoint_files:
            print("❌ No checkpoint files (.pt) found in the repository")
            return
        
        print(f"📋 Found {len(checkpoint_files)} checkpoint file(s):")
        for file in checkpoint_files:
            print(f"  - {file}")
        
        # 各チェックポイントファイルをダウンロード
        for filename in tqdm(checkpoint_files, desc="Downloading checkpoints"):
            try:
                print(f"\n📥 Downloading {filename}...")
                
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    resume_download=True,  # レジューム機能を有効化
                    local_dir_use_symlinks=False  # シンボリックリンクを使わない
                )
                
                # ファイルサイズを取得して表示
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                print(f"✅ Successfully downloaded {filename} ({file_size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
                continue
        
        print(f"\n🎉 Download completed! Files saved to: {local_path.absolute()}")
        
        # ダウンロードされたファイルの一覧を表示
        print("\n📂 Downloaded files:")
        for file in local_path.glob("*.pt"):
            file_size = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({file_size:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Error accessing repository {repo_id}: {e}")
        print("   Please check if the repository exists and is accessible.")

def download_additional_files(local_dir="./checkpoints"):
    """
    チェックポイント以外の重要なファイルもダウンロード（必要に応じて）
    """
    repo_id = "Weibang/NeuroLM"
    additional_files = ["README.md", "config.json"]  # 必要に応じて追加
    
    local_path = Path(local_dir)
    
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id)
    
    for filename in additional_files:
        if filename in repo_files:
            try:
                print(f"📥 Downloading {filename}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")

if __name__ == "__main__":
    # メイン処理
    print("🚀 NeuroLM Checkpoint Downloader")
    print("=" * 50)
    
    # チェックポイントをダウンロード
    download_neurolm_checkpoints()
    
    # 追加ファイルもダウンロード（必要に応じて）
    # download_additional_files()
    
    print("\n✨ All done!")
