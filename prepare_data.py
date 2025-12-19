import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import json
import shutil

class WildDeepfakePreparator:
    def __init__(self, data_root="./WildDeepfake", output_root="./dataset"):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        
        self.select_num = {
            "fake_train": 5,
            "real_train": 5,
            "fake_test": 2,
            "real_test": 2,
        }
        
    def find_tar_files(self, split_name):

        split_dir = self.data_root / "deepfake_in_the_wild" / split_name
        
        if not split_dir.exists():

            split_dir = self.data_root / split_name
        
        if not split_dir.exists():
            print(f"警告: 找不到目录 {split_dir}")
            return []
        
        tar_files = sorted(split_dir.glob("*.tar.gz"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
        return tar_files
    
    def is_already_extracted(self, tar_path):

        video_id = tar_path.stem 
        if video_id.endswith('.tar'):
            video_id = video_id[:-4]  
        
        extracted_dir = tar_path.parent / video_id
        

        if extracted_dir.exists() and any(extracted_dir.iterdir()):
            return True
        return False
    
    def extract_selected(self):

        print("=" * 50)
        print("Step 1: 选择性解压文件")
        print("=" * 50)
        
        extract_info = {}
        
        for split_name, num in self.select_num.items():
            print(f"\n处理 {split_name}...")
            tar_files = self.find_tar_files(split_name)
            
            if not tar_files:
                print(f"  未找到 tar.gz 文件")
                continue
                
            print(f"  找到 {len(tar_files)} 个压缩文件，选取前 {num} 个")
            
            selected = tar_files[:num]
            extract_info[split_name] = {
                "extracted": [],
                "skipped": [],
                "failed": []
            }
            
            for tar_path in tqdm(selected, desc=f"  解压 {split_name}"):

                if self.is_already_extracted(tar_path):
                    extract_info[split_name]["skipped"].append(str(tar_path))
                    continue
                
                try:
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(path=tar_path.parent)
                    extract_info[split_name]["extracted"].append(str(tar_path))
                except Exception as e:
                    print(f"\n  解压失败: {tar_path}, 错误: {e}")
                    extract_info[split_name]["failed"].append(str(tar_path))
            

            info = extract_info[split_name]
            print(f"  新解压: {len(info['extracted'])}, 跳过(已存在): {len(info['skipped'])}, 失败: {len(info['failed'])}")
        
        return extract_info
    
    def build_dataset(self):

        print("\n" + "=" * 50)
        print("Step 2: 构建数据集")
        print("=" * 50)
        
        dataset = {
            "train": [],
            "test": []
        }
        
        # split_name -> (target_split, label, subfolder_name)
        split_mapping = {
            "fake_train": ("train", 1, "fake"),   # 1=fake
            "real_train": ("train", 0, "real"),   # 0=real
            "fake_test": ("test", 1, "fake"),
            "real_test": ("test", 0, "real"),
        }
        
        for split_name, (target_split, label, subfolder) in split_mapping.items():
            print(f"\n扫描 {split_name}...")
            
            split_dir = self.data_root / "deepfake_in_the_wild" / split_name
            if not split_dir.exists():
                split_dir = self.data_root / split_name
            
            if not split_dir.exists():
                print(f"  目录不存在，跳过")
                continue
            
            # 遍历 video_id 目录
            video_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            
            for video_dir in tqdm(video_dirs, desc=f"  处理 {split_name}"):
                video_id = video_dir.name
                
                # 进入 fake/ 或 real/ 子目录
                label_dir = video_dir / subfolder
                if not label_dir.exists():
                    continue
                
                # 遍历 sequence_id 目录
                seq_dirs = [d for d in label_dir.iterdir() if d.is_dir()]
                
                for seq_dir in seq_dirs:
                    seq_id = seq_dir.name
                    
                    # 获取该sequence下的所有帧
                    frames = sorted([
                        f for f in seq_dir.iterdir() 
                        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
                    ], key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
                    
                    if len(frames) == 0:
                        continue
                    
                    # 构建一个样本
                    sample = {
                        "video_id": video_id,
                        "sequence_id": seq_id,
                        "split_name": split_name,
                        "label": label,  # 0=real, 1=fake
                        "num_frames": len(frames),
                        "frames": [str(f) for f in frames],
                        "sequence_dir": str(seq_dir)
                    }
                    
                    dataset[target_split].append(sample)
        
        # 统计信息
        print("\n" + "=" * 50)
        print("数据集统计")
        print("=" * 50)
        
        for split in ["train", "test"]:
            samples = dataset[split]
            if not samples:
                continue
                
            real_count = sum(1 for s in samples if s["label"] == 0)
            fake_count = sum(1 for s in samples if s["label"] == 1)
            total_frames = sum(s["num_frames"] for s in samples)
            avg_frames = total_frames / len(samples) if samples else 0
            
            print(f"\n{split}:")
            print(f"  总序列数: {len(samples)}")
            print(f"  Real: {real_count}, Fake: {fake_count}")
            print(f"  总帧数: {total_frames}")
            print(f"  平均每序列帧数: {avg_frames:.1f}")
        
        return dataset
    
    def save_dataset(self, dataset, filename="dataset.json"):

        output_path = self.output_root / filename
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n数据集索引已保存到: {output_path}")
        return output_path
    
    def run(self):


        self.extract_selected()

        dataset = self.build_dataset()

        self.save_dataset(dataset)
        
        return dataset


if __name__ == "__main__":
    
 
    
    preparator = WildDeepfakePreparator(
        data_root=args.data_root,
        output_root=args.output_root
    )
    
    # 更新解压数量
    preparator.select_num = {
        "fake_train": 40,
        "real_train": 40,
        "fake_test": 10,
        "real_test": 10,
    }
    
    dataset = preparator.run()