import os
import h5py
import time
import faiss
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# === Load Data ===
def read_sift1m(file_name: str = 'SIFT1M.hdf5') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 SIFT1M 数据集，返回 (train, test, gt_top1)。"""
    with h5py.File(file_name, 'r') as f:
        train = f['train'][:]
        test = f['test'][:]
        gt_top1 = f['neighbors'][:, 0]
    return train, test, gt_top1

# === Build HNSW Index ===
def make_hnsw_index(train_vectors: np.ndarray, M: int = 32, ef_construction: int = 200) -> faiss.Index:
    d = train_vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(train_vectors)
    return index

# === Evaluate Latency and Recall ===
def measure_hnsw(index: faiss.Index, test_vectors: np.ndarray, ground_truth: np.ndarray, ef_search: int) -> Tuple[float, float]:
    index.hnsw.efSearch = ef_search
    start = time.perf_counter()
    _, I = index.search(test_vectors, 1)
    end = time.perf_counter()

    recall = float(np.mean(I.ravel() == ground_truth))
    latency_ms = ((end - start) / max(len(test_vectors), 1)) * 1000.0
    return recall, latency_ms

# === Run HNSW Latency vs Recall Benchmark ===
def run_hnsw_benchmark(train: np.ndarray, test: np.ndarray, ground_truth: np.ndarray,
                       ef_values: List[int], M: int = 32, ef_construction: int = 200) -> Tuple[List[float], List[float]]:
    index = make_hnsw_index(train, M, ef_construction)
    recalls: List[float] = []
    latencies: List[float] = []
    for ef in ef_values:
        r, l = measure_hnsw(index, test, ground_truth, ef)
        recalls.append(r)
        latencies.append(l)
        print(f"HNSW efSearch={ef} → Recall@1: {r:.4f}, Latency: {l:.2f} ms")
    return recalls, latencies

# === Placeholder DiskANN Results ===
def load_mock_diskann_results():
    # Replace these with actual results if available
    diskann_recalls = [0.55, 0.70, 0.82, 0.90]
    diskann_latencies = [0.20, 0.35, 0.60, 0.95]  # in ms
    diskann_labels = ["R=20,L=30", "R=30,L=50", "R=40,L=70", "R=60,L=100"]
    return diskann_recalls, diskann_latencies, diskann_labels

# === Plot Latency vs Recall ===
def draw_latency_vs_recall(hnsw_data, diskann_data, ef_vals, diskann_labels):
    hnsw_recalls, hnsw_latencies = hnsw_data
    diskann_recalls, diskann_latencies = diskann_data

    plt.figure(figsize=(8,6))

    plt.plot(hnsw_recalls, hnsw_latencies, marker='o', label='HNSW')
    for i, ef in enumerate(ef_vals):
        plt.annotate(f"ef={ef}", (hnsw_recalls[i], hnsw_latencies[i]), fontsize=9)

    plt.plot(diskann_recalls, diskann_latencies, marker='x', label='DiskANN')
    for i, label in enumerate(diskann_labels):
        plt.annotate(label, (diskann_recalls[i], diskann_latencies[i]), fontsize=9)

    plt.xlabel("1-Recall@1")
    plt.ylabel("Avg Query Latency (ms)")
    plt.title("Latency vs Recall: HNSW vs DiskANN")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency_vs_recall.png")
    plt.show()

# === Main ===
def main():
    # Load dataset
    if not os.path.exists("SIFT1M.hdf5"):
        raise FileNotFoundError("Missing SIFT1M.hdf5 — download it from http://ann-benchmarks.com/sift-128-euclidean.hdf5")

    print("📦 Loading SIFT1M dataset...")
    train, test, neighbors = read_sift1m()

    # Run HNSW
    ef_vals = [10, 50, 100, 200]
    hnsw_data = run_hnsw_benchmark(train, test, neighbors, ef_vals)

    diskann_recalls, diskann_latencies, diskann_labels = load_mock_diskann_results()
    draw_latency_vs_recall(hnsw_data, (diskann_recalls, diskann_latencies), ef_vals, diskann_labels)


if __name__ == "__main__":
    main()