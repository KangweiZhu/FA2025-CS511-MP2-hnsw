import faiss
import h5py
import numpy as np
import os
import requests
from typing import Tuple


def _download_sift1m(h5_path: str):
    """
    从 ann-benchmarks 下载 SIFT1M 数据集
    """
    if os.path.exists(h5_path):
        print(f"数据集已存在: {h5_path}")
        return
    
    print(f"正在下载 SIFT1M 数据集到 {h5_path}...")
    url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(h5_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print(f"\n下载完成: {h5_path}")
    except Exception as e:
        print(f"\n下载失败: {e}")
        print(f"请手动下载数据集并保存为 {h5_path}")
        print(f"下载地址: {url}")
        raise


def _load_sift1m(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 SIFT1M.hdf5 加载数据库向量与查询向量。
    SIFT1M 数据集格式：train (1000000, 128) 为数据库向量，test (10000, 128) 为查询向量。
    """
    assert os.path.exists(h5_path), f"数据文件不存在: {h5_path}"
    with h5py.File(h5_path, "r") as f:
        # SIFT1M 数据集使用 'train' 和 'test' 键
        database = np.array(f['train'][:], dtype=np.float32)
        queries = np.array(f['test'][:], dtype=np.float32)
        return database, queries


def evaluate_hnsw():
    """
    Part 0:
    - 使用 FAISS HNSW 建索引 (M=16, efConstruction=200, efSearch=200)
    - 用第一条查询做检索，取 top-10 邻居索引
    - 将索引逐行写入 output.txt
    """
    h5_path = "./SIFT1M.hdf5"
    output_path = "./output.txt"
    M = 16
    ef_construction = 200
    ef_search = 200
    top_k = 10

    # 如果文件不存在，自动下载
    _download_sift1m(h5_path)
    
    database_embeddings, query_embeddings = _load_sift1m(h5_path)

    num_db, dim = database_embeddings.shape

    # 构建 HNSW 索引（不使用 PQ）
    index = faiss.IndexHNSWFlat(dim, M)
    # 需在 add 之前设置 efConstruction
    index.hnsw.efConstruction = ef_construction
    # 添加数据库向量
    index.add(database_embeddings)
    # 查询阶段设置 efSearch
    index.hnsw.efSearch = ef_search

    # 使用第一条查询向量
    first_query = query_embeddings[0:1]
    distances, indices = index.search(first_query, top_k)
    indices = indices[0].tolist()

    # 写出到 output.txt，每行一个索引
    with open(output_path, "w") as f:
        for idx in indices:
            f.write(f"{int(idx)}\n")


if __name__ == "__main__":
    evaluate_hnsw()