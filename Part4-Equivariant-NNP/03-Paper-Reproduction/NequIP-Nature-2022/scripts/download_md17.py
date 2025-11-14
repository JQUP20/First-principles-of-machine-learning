#!/usr/bin/env python3
"""
下载MD17数据集
Download MD17 Dataset

Usage:
    python download_md17.py --all
    python download_md17.py --molecules aspirin benzene ethanol
"""

import urllib.request
import os
import gzip
import shutil
import argparse
from tqdm import tqdm

# MD17数据集的所有分子
MD17_MOLECULES = [
    'aspirin',
    'benzene',
    'ethanol',
    'malonaldehyde',
    'naphthalene',
    'salicylic_acid',
    'toluene',
    'uracil'
]

# 数据集信息
MOLECULE_INFO = {
    'aspirin': {'atoms': 21, 'configs': 211762},
    'benzene': {'atoms': 12, 'configs': 627983},
    'ethanol': {'atoms': 9, 'configs': 555092},
    'malonaldehyde': {'atoms': 9, 'configs': 993237},
    'naphthalene': {'atoms': 18, 'configs': 326250},
    'salicylic_acid': {'atoms': 16, 'configs': 320231},
    'toluene': {'atoms': 15, 'configs': 442790},
    'uracil': {'atoms': 12, 'configs': 133770}
}

BASE_URL = "http://www.quantum-machine.org/gdml/data/xyz/"

class DownloadProgressBar(tqdm):
    """带进度条的下载器"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_molecule(molecule, data_dir='../data'):
    """
    下载单个分子的MD17数据

    Args:
        molecule: 分子名称
        data_dir: 数据保存目录
    """
    if molecule not in MD17_MOLECULES:
        raise ValueError(f"Unknown molecule: {molecule}. Choose from {MD17_MOLECULES}")

    # 创建目录
    mol_dir = os.path.join(data_dir, f'md17_{molecule}')
    os.makedirs(mol_dir, exist_ok=True)

    # 文件路径
    url = f"{BASE_URL}md17_{molecule}.xyz.gz"
    gz_file = os.path.join(mol_dir, f'{molecule}.xyz.gz')
    xyz_file = os.path.join(mol_dir, f'{molecule}.xyz')

    # 检查是否已存在
    if os.path.exists(xyz_file):
        print(f"✓ {molecule}.xyz already exists, skipping download")
        return xyz_file

    # 下载
    print(f"Downloading {molecule}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=molecule) as t:
            urllib.request.urlretrieve(url, gz_file, reporthook=t.update_to)
    except Exception as e:
        print(f"✗ Error downloading {molecule}: {e}")
        if os.path.exists(gz_file):
            os.remove(gz_file)
        return None

    # 解压
    print(f"Extracting {molecule}...")
    try:
        with gzip.open(gz_file, 'rb') as f_in:
            with open(xyz_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 删除压缩文件
        os.remove(gz_file)

        # 验证
        file_size = os.path.getsize(xyz_file) / (1024**2)  # MB
        info = MOLECULE_INFO[molecule]
        print(f"✓ Successfully downloaded {molecule}")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Atoms: {info['atoms']}, Configurations: {info['configs']:,}")

        return xyz_file

    except Exception as e:
        print(f"✗ Error extracting {molecule}: {e}")
        if os.path.exists(gz_file):
            os.remove(gz_file)
        if os.path.exists(xyz_file):
            os.remove(xyz_file)
        return None

def verify_dataset(xyz_file):
    """验证数据集完整性"""
    try:
        from ase.io import read

        # 读取前几个构型
        atoms_list = read(xyz_file, index=':10')

        if len(atoms_list) != 10:
            return False

        # 检查是否有能量和力
        for atoms in atoms_list:
            if 'energy' not in atoms.info and not hasattr(atoms, 'calc'):
                return False
            if 'forces' not in atoms.arrays:
                return False

        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Download MD17 dataset')
    parser.add_argument('--all', action='store_true', help='Download all molecules')
    parser.add_argument('--molecules', nargs='+', help='Specific molecules to download')
    parser.add_argument('--data-dir', default='../data', help='Data directory')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded datasets')

    args = parser.parse_args()

    # 确定要下载的分子
    if args.all:
        molecules = MD17_MOLECULES
    elif args.molecules:
        molecules = args.molecules
    else:
        print("Please specify --all or --molecules")
        parser.print_help()
        return

    print("=" * 60)
    print("MD17 Dataset Downloader")
    print("=" * 60)
    print(f"Molecules to download: {', '.join(molecules)}")
    print(f"Data directory: {args.data_dir}")
    print()

    # 下载
    successful = []
    failed = []

    for molecule in molecules:
        try:
            xyz_file = download_molecule(molecule, args.data_dir)

            if xyz_file and os.path.exists(xyz_file):
                # 验证
                if args.verify:
                    print(f"Verifying {molecule}...")
                    if verify_dataset(xyz_file):
                        print(f"✓ Verification passed")
                        successful.append(molecule)
                    else:
                        print(f"✗ Verification failed")
                        failed.append(molecule)
                else:
                    successful.append(molecule)
            else:
                failed.append(molecule)

        except Exception as e:
            print(f"✗ Error with {molecule}: {e}")
            failed.append(molecule)

        print()

    # 总结
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(molecules)}")
    if successful:
        print(f"  {', '.join(successful)}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(molecules)}")
        print(f"  {', '.join(failed)}")

    # 总大小
    total_size = 0
    for molecule in successful:
        xyz_file = os.path.join(args.data_dir, f'md17_{molecule}', f'{molecule}.xyz')
        if os.path.exists(xyz_file):
            total_size += os.path.getsize(xyz_file)

    print(f"\nTotal size: {total_size / (1024**3):.2f} GB")
    print("\n✓ Download completed!")

if __name__ == '__main__':
    main()
