import os
import h5py
import time
import faiss
import requests
import numpy as np
import matplotlib.pyplot as plt
import csv
import logging
from typing import Tuple, Dict, List


# 数据集URL和名称配置
DATASETS = [
    {
        'name': 'coco-t2i',
        'url': 'https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-t2i-512-angular.hdf5',
        'metric': 'angular'
    },
    {
        'name': 'coco-i2i',
        'url': 'https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5',
        'metric': 'angular'
    },
    {
        'name': 'lastfm',
        'url': 'http://ann-benchmarks.com/lastfm-64-dot.hdf5',
        'metric': 'angular'
    },
    {
        'name': 'mnist',
        'url': 'http://ann-benchmarks.com/mnist-784-euclidean.hdf5',
        'metric': 'euclidean'
    }
]

M_VALUES = [4, 8, 12, 24, 48]
EF_CONSTRUCTION = 200
EF_SEARCH = 200


# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)


def _get_output_dir() -> str:
    try:
        return os.path.dirname(__file__)
    except NameError:
        return '.'


def download_dataset(url: str, filename: str):
    """下载数据集文件（带进度条）"""
    filepath = os.path.join('.', filename)
    
    if os.path.exists(filepath):
        logging.info(f"文件已存在: {filepath}")
        return filepath
    
    logging.info(f"开始下载: {filename}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print()
        logging.info(f"下载完成: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"下载失败: {e}")
        logging.error(f"请手动下载数据集并保存为 {filepath}")
        logging.error(f"下载地址: {url}")
        raise


def load_data_from_url(name: str, url: str, metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    从URL加载数据集
    返回: (train, test, neighbors, metric)
    """
    filename = url.split("/")[-1]
    filepath = download_dataset(url, filename)
    
    with h5py.File(filepath, 'r') as f:
        logging.info(f"数据集 {name} 键: {list(f.keys())}")
        train = f['train'][:].astype(np.float32)
        test = f['test'][:].astype(np.float32)
        neighbors = f['neighbors'][:]  # Ground truth 最近邻
    
    logging.info(f"训练集: {train.shape}, 测试集: {test.shape}, GT: {neighbors.shape}")
    return train, test, neighbors, metric


def build_hnsw_index(train_data: np.ndarray, M: int, ef_construction: int, metric: str) -> Tuple[faiss.Index, float]:
    """
    构建 HNSW 索引
    返回: (index, build_time)
    """
    d = train_data.shape[1]
    
    # 根据距离类型选择合适的索引
    if metric == 'angular':
        # Angular 距离使用内积（需要归一化）
        train_data = train_data.copy()
        faiss.normalize_L2(train_data)
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    elif metric == 'euclidean':
        # 欧氏距离使用 L2
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    else:
        # 默认使用 L2
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    
    index.hnsw.efConstruction = ef_construction
    
    # 构建索引
    start = time.time()
    index.add(train_data)
    build_time = time.time() - start
    
    return index, build_time


def evaluate(index: faiss.Index, test_data: np.ndarray, ground_truth: np.ndarray, 
             ef_search: int, metric: str) -> Tuple[float, float]:
    """
    评估索引性能
    返回: (recall, qps)
    """
    index.hnsw.efSearch = ef_search
    
    # 对于 angular 距离，需要归一化查询向量
    if metric == 'angular':
        test_data = test_data.copy()
        faiss.normalize_L2(test_data)
    
    # 执行查询
    start = time.time()
    D, I = index.search(test_data, 1)
    query_time = time.time() - start
    
    # 计算 Recall@1
    correct = (I[:, 0] == ground_truth[:, 0]).sum()
    recall = correct / len(test_data)
    
    # 计算 QPS
    qps = len(test_data) / query_time
    
    return recall, qps


def run_experiment(name: str, url: str, metric: str) -> Dict:
    """
    对单个数据集运行实验
    返回结果字典
    """
    logging.info("="*60)
    logging.info(f"实验: {name} (metric={metric})")
    logging.info("="*60)
    
    # 加载数据
    train, test, gt, metric = load_data_from_url(name, url, metric)
    
    results = {
        "name": name,
        "train_size": int(train.shape[0]),
        "M": [],
        "recall": [],
        "qps": [],
        "build_time": [],
        "1_recall": []
    }
    
    # 对每个 M 值进行测试
    for M in M_VALUES:
        logging.info(f"开始构建索引: M={M}, efConstruction={EF_CONSTRUCTION}")
        
        # 构建索引
        t0 = time.time()
        index, build_time = build_hnsw_index(train.copy(), M, EF_CONSTRUCTION, metric)
        logging.info(f"索引构建完成，用时 {build_time:.2f}s")
        
        # 评估性能
        recall, qps = evaluate(index, test.copy(), gt, EF_SEARCH, metric)
        one_minus_recall = 1.0 - recall
        logging.info(f"评估: Recall@1={recall:.4f}, 1-Recall@1={one_minus_recall:.4f}, QPS={qps:.2f}")
        
        # 记录结果
        results["M"].append(M)
        results["recall"].append(recall)
        results["qps"].append(qps)
        results["build_time"].append(build_time)
        results["1_recall"].append(one_minus_recall)
    
    return results


def plot_metric(results: Dict[str, Dict], xlabel: str, ylabel: str, 
                metric_name: str, y_key: str, output_filename: str):
    """
    绘制指标图表
    """
    plt.figure(figsize=(10, 6))
    
    for name in results:
        x = results[name]["recall"]
        y = results[name][y_key]
        M_list = results[name]["M"]
        train_size = results[name].get("train_size")
        label = f"{name} (N={train_size})" if train_size is not None else name

        plt.plot(x, y, marker='o', label=label, linewidth=2, markersize=8)
        
        # 为每个点添加 M 值标注
        for i, M in enumerate(M_list):
            plt.text(x[i], y[i], f"  M={M}", fontsize=8, alpha=0.7)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} vs {xlabel}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = _get_output_dir()
    output_path = os.path.join(out_dir, os.path.basename(output_filename))
    plt.savefig(output_path, dpi=300)
    logging.info(f"图表已保存: {output_path}")
    plt.close()


def save_results_to_csv(results: Dict[str, Dict], output_filename: str):
    """
    保存结果到 CSV 文件
    """
    # 准备数据行
    rows = []
    for name in results:
        for i in range(len(results[name]["M"])):
            rows.append({
                'dataset': name,
                'M': results[name]["M"][i],
                'recall@1': results[name]["recall"][i],
                '1_recall@1': results[name]["1_recall"][i],
                'qps': results[name]["qps"][i],
                'build_time': results[name]["build_time"][i]
            })
    
    # 写入 CSV
    if rows:
        fieldnames = list(rows[0].keys())
        out_dir = _get_output_dir()
        output_path = os.path.join(out_dir, os.path.basename(output_filename))
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"结果已保存到: {output_path}")


def print_summary_table(results: Dict[str, Dict]):
    """
    打印结果汇总表格
    """
    logging.info("结果汇总")
    
    # 打印表头
    header = f"{'Dataset':<15} | {'M':>5} | {'Recall@1':>10} | {'1-Recall@1':>12} | {'QPS':>10} | {'Build Time':>12}"
    print("\n" + header)
    print("-" * len(header))
    
    # 打印数据
    for name in results:
        for i in range(len(results[name]["M"])):
            row = (f"{name:<15} | "
                   f"{results[name]['M'][i]:>5} | "
                   f"{results[name]['recall'][i]:>10.4f} | "
                   f"{results[name]['1_recall'][i]:>12.4f} | "
                   f"{results[name]['qps'][i]:>10.2f} | "
                   f"{results[name]['build_time'][i]:>12.2f}s")
            print(row)


def generate_report(results: Dict[str, Dict]):
    out_dir = _get_output_dir()
    report_path = os.path.join(out_dir, "part2_report.md")
    qps_img = "part2_qps_vs_recall.png"
    build_img = "part2_build_time_vs_recall.png"

    lines: List[str] = []
    lines.append("## Part 2 报告：HNSW 在不同数据集大小上的权衡\n")
    lines.append("")
    lines.append("### 数据集规模\n")
    for name in results:
        n = results[name].get("train_size")
        lines.append(f"- {name}: N={n}")
    lines.append("")
    lines.append("### 图1：QPS vs Recall@1\n")
    lines.append(f"![QPS vs Recall]({qps_img})\n")
    lines.append("- 横轴：Recall@1；纵轴：QPS。不同曲线代表不同数据集规模，点上标注为对应的 M 值。\n")
    lines.append("")
    lines.append("### 图2：Index Build Time vs Recall@1\n")
    lines.append(f"![Build Time vs Recall]({build_img})\n")
    lines.append("- 横轴：Recall@1；纵轴：索引构建时间（秒）。不同曲线代表不同数据集规模，点上标注为对应的 M 值。\n")
    lines.append("")
    lines.append("### 简要分析\n")
    lines.append("- 随 M 增大，Recall 提升但构建时间与查询代价上升，QPS 通常下降。\n")
    lines.append("- 数据集越大，同等 M 下 Recall 较低且构建时间更长；需要更大的 M 才能达到相似 Recall。\n")
    lines.append("- 需要在目标 Recall 与可接受吞吐/构建开销之间权衡，选择合适的 M。\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logging.info(f"报告已生成: {report_path}")


def main():
    """主函数"""
    logging.info("Part 2: HNSW Benchmarking with Increasing Dataset Sizes")
    
    # 运行所有数据集的实验
    all_results = {}
    
    for dataset_info in DATASETS:
        try:
            name = dataset_info['name']
            url = dataset_info['url']
            metric = dataset_info['metric']
            
            results = run_experiment(name, url, metric)
            all_results[name] = results
            
        except Exception as e:
            logging.error(f"处理数据集 {dataset_info['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成可视化结果
    if all_results:
        logging.info("生成可视化结果与报告…")
        
        # 图1: QPS vs Recall@1 (用不同曲线表示不同数据集，标注M值)
        plot_metric(all_results, "Recall@1", "QPS", "qps", "qps", 
                   "part2_qps_vs_recall.png")
        
        # 图2: Index Build Time vs Recall@1 (用不同曲线表示不同数据集，标注M值)
        plot_metric(all_results, "Recall@1", "Index Build Time (s)", 
                   "build_time", "build_time", "part2_build_time_vs_recall.png")
        
        # 保存 CSV 结果
        save_results_to_csv(all_results, "part2_results.csv")
        
        # 打印汇总表格
        print_summary_table(all_results)

        # 生成报告
        generate_report(all_results)
        logging.info("所有实验完成！")
    else:
        logging.error("没有结果可显示！")


if __name__ == "__main__":
    main()
