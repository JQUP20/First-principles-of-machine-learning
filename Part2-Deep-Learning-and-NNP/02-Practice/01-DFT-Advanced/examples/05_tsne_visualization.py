#!/usr/bin/env python3
"""
t-SNE数据集可视化
t-SNE Dataset Visualization

使用t-SNE降维技术可视化AIMD数据集的结构多样性
Using t-SNE dimensionality reduction to visualize structural diversity of AIMD dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ase.io import read, Trajectory


def compute_descriptors(atoms_list, rcut=5.0):
    """
    计算简单的结构描述符（径向分布函数）
    Compute simple structural descriptors (radial distribution function)

    Parameters:
    -----------
    atoms_list : list of ASE Atoms
        原子结构列表
    rcut : float
        截断半径 (Angstrom)

    Returns:
    --------
    descriptors : ndarray
        描述符矩阵 (n_structures, n_features)
    """
    from scipy.spatial import distance_matrix

    print("计算结构描述符...")
    print("Computing structural descriptors...")

    descriptors = []

    # RDF参数
    nbins = 50
    dr = rcut / nbins
    bins = np.linspace(0, rcut, nbins + 1)

    for i, atoms in enumerate(atoms_list):
        if (i + 1) % 10 == 0:
            print(f"  处理 {i+1}/{len(atoms_list)}")

        # 计算所有原子对的距离
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        # 简化：不考虑周期性边界条件
        # 完整实现应使用atoms.get_all_distances(mic=True)
        dists = distance_matrix(positions, positions)

        # 计算径向分布函数
        rdf, _ = np.histogram(dists.flatten(), bins=bins)

        # 归一化
        rdf = rdf.astype(float)
        rdf /= (rdf.sum() + 1e-10)  # 避免除零

        descriptors.append(rdf)

    descriptors = np.array(descriptors)

    print(f"描述符形状: {descriptors.shape}")
    print(f"Descriptor shape: {descriptors.shape}")

    return descriptors


def run_tsne(descriptors, perplexity=30, n_iter=1000):
    """
    运行t-SNE降维
    Run t-SNE dimensionality reduction

    Parameters:
    -----------
    descriptors : ndarray
        描述符矩阵
    perplexity : float
        t-SNE perplexity参数
    n_iter : int
        迭代次数

    Returns:
    --------
    embedding : ndarray
        二维嵌入 (n_structures, 2)
    """
    print("\n运行t-SNE降维...")
    print("Running t-SNE...")

    # 标准化
    scaler = StandardScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        verbose=1
    )

    embedding = tsne.fit_transform(descriptors_scaled)

    print(f"t-SNE完成！")
    print(f"t-SNE completed!")

    return embedding


def visualize_tsne(embedding, energies=None, temperatures=None,
                  output_file='tsne_visualization.png'):
    """
    可视化t-SNE结果
    Visualize t-SNE results

    Parameters:
    -----------
    embedding : ndarray
        t-SNE嵌入
    energies : ndarray, optional
        能量数据（用于颜色编码）
    temperatures : ndarray, optional
        温度数据（用于颜色编码）
    output_file : str
        输出文件名
    """
    print("\n绘制t-SNE图...")
    print("Plotting t-SNE...")

    if energies is not None and temperatures is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 按能量着色
        scatter1 = ax1.scatter(
            embedding[:, 0], embedding[:, 1],
            c=energies,
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.set_title('Colored by Energy', fontsize=14, fontweight='bold')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Energy (eV)', fontsize=12)

        # 按温度着色
        scatter2 = ax2.scatter(
            embedding[:, 0], embedding[:, 1],
            c=temperatures,
            cmap='coolwarm',
            s=50,
            alpha=0.7
        )
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax2.set_title('Colored by Temperature', fontsize=14, fontweight='bold')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Temperature (K)', fontsize=12)

    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=range(len(embedding)),
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE Visualization of AIMD Dataset',
                    fontsize=14, fontweight='bold')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Time Step', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"t-SNE图已保存: {output_file}")
    print(f"t-SNE plot saved: {output_file}")
    plt.show()


def analyze_clusters(embedding, n_clusters=3):
    """
    聚类分析
    Cluster analysis

    Parameters:
    -----------
    embedding : ndarray
        t-SNE嵌入
    n_clusters : int
        聚类数量
    """
    from sklearn.cluster import KMeans

    print(f"\n运行K-Means聚类 (k={n_clusters})...")
    print(f"Running K-Means clustering (k={n_clusters})...")

    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    print(f"聚类完成！")
    print(f"Clustering completed!")

    # 统计每个簇的样本数
    for i in range(n_clusters):
        count = np.sum(labels == i)
        print(f"  簇 {i}: {count} 个样本")
        print(f"  Cluster {i}: {count} samples")

    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels,
        cmap='tab10',
        s=50,
        alpha=0.7
    )
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f't-SNE with K-Means Clustering (k={n_clusters})',
             fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('tsne_clusters.png', dpi=300)
    print("\n聚类图已保存: tsne_clusters.png")
    print("Cluster plot saved: tsne_clusters.png")
    plt.show()

    return labels


def main():
    """
    主函数
    Main function
    """
    print("\n" + "=" * 60)
    print("t-SNE数据集可视化")
    print("t-SNE Dataset Visualization")
    print("=" * 60)

    # 检查是否有轨迹文件
    try:
        # 读取AIMD轨迹
        print("\n读取AIMD轨迹...")
        print("Loading AIMD trajectory...")
        traj = Trajectory('aimd_trajectory.traj', 'r')

        # 采样（避免过多数据点）
        skip = max(1, len(traj) // 100)  # 最多100个点
        atoms_list = [atoms for atoms in traj[::skip]]

        print(f"加载了 {len(atoms_list)} 个构型")
        print(f"Loaded {len(atoms_list)} configurations")

    except FileNotFoundError:
        print("\n错误：未找到 aimd_trajectory.traj 文件")
        print("Error: aimd_trajectory.traj not found")
        print("请先运行 04_aimd_dataset.py 生成数据")
        print("Please run 04_aimd_dataset.py first to generate data")
        return

    # 提取能量和温度信息
    from ase import units
    energies = []
    temperatures = []

    for atoms in atoms_list:
        energies.append(atoms.get_potential_energy())
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))
        temperatures.append(temp)

    energies = np.array(energies)
    temperatures = np.array(temperatures)

    # 1. 计算描述符
    print("\n步骤 1: 计算结构描述符")
    print("Step 1: Compute structural descriptors")
    descriptors = compute_descriptors(atoms_list, rcut=5.0)

    # 2. 运行t-SNE
    print("\n步骤 2: t-SNE降维")
    print("Step 2: t-SNE dimensionality reduction")
    embedding = run_tsne(descriptors, perplexity=30, n_iter=1000)

    # 3. 可视化
    print("\n步骤 3: 可视化t-SNE结果")
    print("Step 3: Visualize t-SNE results")
    visualize_tsne(
        embedding,
        energies=energies,
        temperatures=temperatures,
        output_file='tsne_visualization.png'
    )

    # 4. 聚类分析
    print("\n步骤 4: 聚类分析")
    print("Step 4: Cluster analysis")
    labels = analyze_clusters(embedding, n_clusters=3)

    # 5. 统计分析
    print("\n" + "=" * 60)
    print("数据集统计")
    print("Dataset Statistics")
    print("=" * 60)
    print(f"构型总数: {len(atoms_list)}")
    print(f"Total configurations: {len(atoms_list)}")
    print(f"能量范围: {energies.min():.4f} - {energies.max():.4f} eV")
    print(f"Energy range: {energies.min():.4f} - {energies.max():.4f} eV")
    print(f"温度范围: {temperatures.min():.1f} - {temperatures.max():.1f} K")
    print(f"Temperature range: {temperatures.min():.1f} - {temperatures.max():.1f} K")

    print("\n" + "=" * 60)
    print("完成！")
    print("Completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
