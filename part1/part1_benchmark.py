import faiss
import h5py
import numpy as np
import os
import time
import requests
import matplotlib.pyplot as plt
import csv
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
    返回：
      - database_embeddings: (N, d) float32 - train embeddings
      - query_embeddings: (Q, d) float32 - test/query embeddings
    """
    assert os.path.exists(h5_path), f"数据文件不存在: {h5_path}"
    with h5py.File(h5_path, "r") as f:
        # SIFT1M 数据集使用 'train' 和 'test' 键
        database = np.array(f['train'][:], dtype=np.float32)
        queries = np.array(f['test'][:], dtype=np.float32)
        return database, queries


def benchmark_hnsw_vs_lsh():
    """
    Part 1:
    - 对比 HNSW 与 LSH 的 Recall@1 与 QPS
    - HNSW: M=32, efSearch ∈ [10, 50, 100, 200]
    - LSH: nbits ∈ [32, 64, 512, 768]
    - 绘制 QPS vs Recall 曲线并保存
    """
    h5_path = "./SIFT1M.hdf5"
    output_plot_path = "./part1_qps_vs_recall.png"

    # 如果文件不存在，自动下载
    _download_sift1m(h5_path)
    
    print("加载数据集...")
    database_embeddings, query_embeddings = _load_sift1m(h5_path)
    
    num_db, dim = database_embeddings.shape
    num_queries = query_embeddings.shape[0]
    print(f"数据库向量: {num_db} 个, 维度: {dim}")
    print(f"查询向量: {num_queries} 个")

    print("计算精确最近邻用于 Recall@1 计算...")
    # 精确 NN（用于计算 Recall@1）
    exact_index = faiss.IndexFlatL2(dim)
    exact_index.add(database_embeddings)
    exact_dists, exact_ids = exact_index.search(query_embeddings, 1)
    exact_top1 = exact_ids.reshape(-1)

    # HNSW 配置: M=32, efSearch ∈ [10, 50, 100, 200]
    print("\n测试 HNSW (M=32)...")
    hnsw_M = 32
    hnsw_ef_search_list = [10, 50, 100, 200]
    hnsw_points = []  # (recall, qps)

    # 预构建一次索引（M=32, efConstruction=200）
    hnsw_index = faiss.IndexHNSWFlat(dim, hnsw_M)
    hnsw_index.hnsw.efConstruction = 200
    print("构建 HNSW 索引中...")
    hnsw_index.add(database_embeddings)

    for ef_s in hnsw_ef_search_list:
        hnsw_index.hnsw.efSearch = ef_s
        t0 = time.time()
        _, ann_ids = hnsw_index.search(query_embeddings, 1)
        elapsed = time.time() - t0
        qps = num_queries / max(elapsed, 1e-9)
        recall_at_1 = float(np.mean((ann_ids.reshape(-1) == exact_top1)))
        hnsw_points.append((recall_at_1, qps))
        print(f"  efSearch={ef_s}: Recall@1={recall_at_1:.4f}, QPS={qps:.2f}")

    # LSH 配置: nbits ∈ [32, 64, 512, 768]
    print("\n测试 LSH...")
    lsh_nbits_list = [32, 64, 512, 768]
    lsh_points = []  # (recall, qps)
    
    for nbits in lsh_nbits_list:
        print(f"  构建 LSH 索引 (nbits={nbits})...")
        lsh_index = faiss.IndexLSH(dim, nbits)
        lsh_index.add(database_embeddings)
        t0 = time.time()
        _, ann_ids = lsh_index.search(query_embeddings, 1)
        elapsed = time.time() - t0
        qps = num_queries / max(elapsed, 1e-9)
        recall_at_1 = float(np.mean((ann_ids.reshape(-1) == exact_top1)))
        lsh_points.append((recall_at_1, qps))
        print(f"  nbits={nbits}: Recall@1={recall_at_1:.4f}, QPS={qps:.2f}")

    # 仅添加数据收集：将结果保存为 CSV 到 part1 目录
    try:
        output_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'

        # HNSW 结果 CSV
        hnsw_csv_path = os.path.join(output_dir, "part1_hnsw_results.csv")
        with open(hnsw_csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["algorithm", "M", "efSearch", "recall@1", "1-recall@1", "qps"])
            for (recall, qps), ef_s in zip(hnsw_points, hnsw_ef_search_list):
                writer.writerow(["HNSW", hnsw_M, ef_s, f"{recall:.6f}", f"{1.0 - recall:.6f}", f"{qps:.6f}"])
        print(f"HNSW 结果已保存: {hnsw_csv_path}")

        # LSH 结果 CSV
        lsh_csv_path = os.path.join(output_dir, "part1_lsh_results.csv")
        with open(lsh_csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["algorithm", "nbits", "recall@1", "1-recall@1", "qps"])
            for (recall, qps), nb in zip(lsh_points, lsh_nbits_list):
                writer.writerow(["LSH", nb, f"{recall:.6f}", f"{1.0 - recall:.6f}", f"{qps:.6f}"])
        print(f"LSH 结果已保存: {lsh_csv_path}")

        # 合并结果 CSV
        combined_csv_path = os.path.join(output_dir, "part1_results.csv")
        with open(combined_csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["algorithm", "param", "value", "M", "recall@1", "1-recall@1", "qps"])
            for (recall, qps), ef_s in zip(hnsw_points, hnsw_ef_search_list):
                writer.writerow(["HNSW", "efSearch", ef_s, hnsw_M, f"{recall:.6f}", f"{1.0 - recall:.6f}", f"{qps:.6f}"])
            for (recall, qps), nb in zip(lsh_points, lsh_nbits_list):
                writer.writerow(["LSH", "nbits", nb, "", f"{recall:.6f}", f"{1.0 - recall:.6f}", f"{qps:.6f}"])
        print(f"综合结果已保存: {combined_csv_path}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")

    # 绘图：QPS vs Recall
    print("\n绘制图表...")
    plt.figure(figsize=(10, 6))
    if hnsw_points:
        hnsw_recalls = [r for r, _ in hnsw_points]
        hnsw_qps = [q for _, q in hnsw_points]
        plt.plot(hnsw_recalls, hnsw_qps, "-o", label="HNSW (M=32)", linewidth=2, markersize=8)
        for (r, q), ef_s in zip(hnsw_points, hnsw_ef_search_list):
            plt.annotate(f"ef={ef_s}", (r, q), xytext=(5, 5), textcoords='offset points', fontsize=9)
    if lsh_points:
        lsh_recalls = [r for r, _ in lsh_points]
        lsh_qps = [q for _, q in lsh_points]
        plt.plot(lsh_recalls, lsh_qps, "-s", label="LSH", linewidth=2, markersize=8)
        for (r, q), nb in zip(lsh_points, lsh_nbits_list):
            plt.annotate(f"nbits={nb}", (r, q), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel("Recall@1", fontsize=12)
    plt.ylabel("QPS (Queries Per Second)", fontsize=12)
    plt.title("QPS vs Recall@1: HNSW vs LSH (SIFT1M)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"图表已保存到: {output_plot_path}")


if __name__ == "__main__":
    benchmark_hnsw_vs_lsh()
