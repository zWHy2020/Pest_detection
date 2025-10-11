# data/prepare_data.py
"""
数据准备脚本
将原始数据转换为标准格式
"""

import os
import json
import shutil
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

def create_dummy_hsi(rgb_path: str, hsi_channels: int = 224, hsi_size: int = 64) -> np.ndarray:
    """
    创建虚拟高光谱数据（用于测试）
    实际应用中应该使用真实的高光谱数据
    """
    # 加载RGB图像
    rgb_img = Image.open(rgb_path).convert('RGB')
    rgb_img = rgb_img.resize((hsi_size, hsi_size))
    rgb_array = np.array(rgb_img)
    
    # 创建虚拟HSI数据
    hsi = np.random.randn(hsi_channels, hsi_size, hsi_size).astype(np.float32)
    
    # 让前3个波段接近RGB
    hsi[0] = rgb_array[:, :, 0] / 255.0
    hsi[1] = rgb_array[:, :, 1] / 255.0
    hsi[2] = rgb_array[:, :, 2] / 255.0
    
    # 归一化
    hsi = (hsi - hsi.mean()) / (hsi.std() + 1e-8)
    
    return hsi

def generate_text_description(class_name: str, image_name: str = "") -> str:
    """
    根据类别生成文本描述
    实际应用中应该使用真实的专家标注
    """
    descriptions = {
        "Pepper__bell___Bacterial_spot": ["叶脉附近出现了极其微小的、针尖状的早期侵染斑点。", "叶片上出现了数个不规则的褐色坏死病斑，部分病斑中心颜色较浅，边缘呈深褐色。", "叶片上布满了大量细小的、深褐色的坏死斑点，并伴随着大面积的黄化现象。", "这些可能是细菌性叶斑病的症状。"],
        "Pepper__bell___healthy": ["叶片呈鲜亮的翠绿色。", "主侧叶脉纹理清晰，无任何病斑或损伤迹象。", "这应该是健康的辣椒叶片。"],
        "Potato___Early_blight": ["叶片上出现深褐色、近圆形的病斑,病斑内部具有一圈圈颜色深浅交替的同心轮纹，呈现出典型的“靶斑”症状。", "病斑周围的叶片组织明显失绿变黄，形成清晰的黄色晕圈，且多个病斑可融合成不规则的大块坏死区域。", "这应该是早疫病的症状。"],
        "Potato___healthy": ["叶片颜色为深绿色，整体轮廓通常为卵圆形，边缘没有尖锐的锯齿，显得较为圆润。", "叶片表面并非完全光滑，而是呈现出明显的、凹凸不平的褶皱感，叶脉下陷，叶肉部分向上隆起。", "这应该是健康的马铃薯叶片。"],
        "Potato___Late_blight": ["叶片上出现了大面积、形状不规则的坏死病斑，", "病斑颜色从浅褐色到深黑褐色过渡，病斑边缘模糊不清，呈现水渍状的暗绿色。", "这是晚疫病的典型症状。"],
        "Tomato__Target_Spot": ["叶片上散布着多个细小的、深褐色至黑色的坏死病斑。", "病斑中心有明显的坏死点，外观类似针头戳刺的痕迹。", "病斑通常较小,直径多在1-2毫米,但数量众多,密集分布于叶片表面。", "这是靶斑病的典型症状。"],
        "Tomato__Tomato_mosaic_virus": ["叶片系统性变形，叶面皱缩、凹凸不平，出现“泡状”的隆起。", "叶片呈现出典型的花叶症状，叶色黄绿驳杂、深浅不一。", "叶片边缘向上或向下卷曲，整体形态不舒展", "这是花叶病毒的典型症状。"],
        "Tomato__Tomato_YellowLeaf__Curl_Virus": ["叶片边缘普遍向上卷曲,整个叶片呈现出勺状或杯状的畸形外观。", "叶片失绿黄化，尤其是在叶片边缘和叶脉间区域。", "叶片显著小于正常叶片，叶面严重皱缩不平。", "这是黄化曲叶病毒的典型症状。"],
        "Tomato_Bacterial_spot": ["叶片上形成针尖大小、水渍状的暗色斑点", "叶片病斑具有油腻感，周围常伴随一圈黄色的晕圈。", "叶片大面积黄化、干枯和早落。", "这是细菌性叶斑病的典型症状。"],
        "Tomato_Early_blight": ["复叶上，出现了典型的“靶斑”症状，病斑呈深褐色圆形，内部具有一圈圈颜色深浅交替的同心轮纹。", "多个独立的“靶斑”在叶片上融合，形成大面积不规则的干枯坏死区域，叶片边缘焦枯卷曲。", "这是早疫病的典型症状。"],
        "Tomato_healthy": ["叶片呈鲜亮的绿色，形态为典型的羽状复叶。", "叶片表面覆盖着一层细密的腺毛，使其在光线下看起来有毛茸茸的质感。", "叶片舒展平整，叶脉清晰", "这是健康的叶片。"],
        "Tomato_Late_blight": ["叶片上，出现了大面积、形状不规则的灰绿色水渍状病斑。", "叶片背面会出现一层标志性的白色霜状霉层。", "病斑没有同心轮纹，而是呈现为大片污秽状的坏死区域。", "这是典型的晚疫病症状。"],
        "Tomato_Leaf_Mold": ["叶片正面（上表面）出现了边界不清晰的、淡绿色至鲜黄色的褪绿斑块。", "黄色斑块主要在叶脉之间形成，形状不规则", "正面黄斑相对应的叶片背面，会生成一层天鹅绒质感的、橄榄绿色至深褐色的霉层。", "这是叶霉病的典型症状。"],
        "Tomato_Septoria_leaf_spot":["叶片上散布着大量细小的、近圆形的病斑。", "病斑中心为灰白色或浅褐色，边缘则为深褐色或黑色。", "病斑的灰白色中心，有时可见许多针尖大小的黑色小点。", "这是典型的斑枯病症状。"],
        "Tomato_Spider_mites_Two_spotted_spider_mite": ["叶片上布满了大量密集的、针尖大小的黄白色小斑点。", "叶片整体失去光泽，叶面呈现出斑驳的灰黄色或青铜色。", "在叶脉附近，会发现细密的蛛网", "这是典型的二斑叶螨虫害症状。"]
    }
    
    base_description = descriptions.get(
        class_name.lower(),
        f"病虫害类别: {class_name}"
    )
    
    return base_description

class DatasetPreparer:
    """数据集准备器"""
    
    def __init__(
        self,
        rgb_folder: str,
        output_folder: str,
        hsi_folder: Optional[str] = None,
        text_file: Optional[str] = None,
        create_dummy_hsi: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        self.rgb_folder = rgb_folder
        self.output_folder = output_folder
        self.hsi_folder = hsi_folder
        self.text_file = text_file
        self.create_dummy_hsi = create_dummy_hsi
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 加载文本描述（如果有）
        self.text_descriptions = {}
        if text_file and os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                self.text_descriptions = json.load(f)
    
    def prepare(self):
        """执行数据准备"""
        print("=" * 60)
        print("开始准备数据集")
        print("=" * 60)
        
        # 创建输出目录
        os.makedirs(self.output_folder, exist_ok=True)
        for folder in ['rgb', 'hsi']:
            os.makedirs(os.path.join(self.output_folder, folder), exist_ok=True)
        
        # 收集所有样本
        print("\n1. 收集样本...")
        samples, class_to_idx = self._collect_samples()
        print(f"   找到 {len(samples)} 个样本，{len(class_to_idx)} 个类别")
        
        # 划分数据集
        print("\n2. 划分数据集...")
        train_samples, val_samples, test_samples = self._split_dataset(samples)
        print(f"   训练集: {len(train_samples)} 样本")
        print(f"   验证集: {len(val_samples)} 样本")
        print(f"   测试集: {len(test_samples)} 样本")
        
        # 处理并保存数据
        print("\n3. 处理数据...")
        self._process_and_save(train_samples, val_samples, test_samples, class_to_idx)
        
        # 保存类别映射
        print("\n4. 保存类别映射...")
        with open(os.path.join(self.output_folder, 'class_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("数据准备完成！")
        print(f"输出目录: {self.output_folder}")
        print("=" * 60)
    
    def _collect_samples(self) -> Tuple[List[Dict], Dict[str, int]]:
        """收集所有样本"""
        samples = []
        class_names = []
        
        # 遍历RGB文件夹
        for class_name in os.listdir(self.rgb_folder):
            class_path = os.path.join(self.rgb_folder, class_name)
            if not os.path.isdir(class_path):
                continue
            
            class_names.append(class_name)
            
            for img_file in os.listdir(class_path):
                if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                base_name = os.path.splitext(img_file)[0]
                rgb_path = os.path.join(class_path, img_file)
                
                # HSI路径
                if self.hsi_folder:
                    hsi_path = os.path.join(self.hsi_folder, class_name, f"{base_name}.npy")
                else:
                    hsi_path = None
                
                # 文本描述
                description = self.text_descriptions.get(
                    base_name,
                    generate_text_description(class_name, base_name)
                )
                
                samples.append({
                    'id': f"{class_name}_{base_name}",
                    'rgb_source': rgb_path,
                    'hsi_source': hsi_path,
                    'class': class_name,
                    'description': description
                })
        
        # 创建类别映射
        class_to_idx = {name: idx for idx, name in enumerate(sorted(set(class_names)))}
        
        return samples, class_to_idx
    
    def _split_dataset(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """划分数据集"""
        # 按类别分层划分
        labels = [s['class'] for s in samples]
        
        # 训练集和临时集
        train_samples, temp_samples = train_test_split(
            samples,
            train_size=self.train_ratio,
            stratify=labels,
            random_state=42
        )
        
        # 验证集和测试集
        temp_labels = [s['class'] for s in temp_samples]
        val_samples, test_samples = train_test_split(
            temp_samples,
            train_size=self.val_ratio / (self.val_ratio + self.test_ratio),
            stratify=temp_labels,
            random_state=42
        )
        
        return train_samples, val_samples, test_samples
    
    def _process_and_save(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        class_to_idx: Dict[str, int]
    ):
        """处理并保存数据"""
        
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        for split_name, samples in splits.items():
            print(f"\n   处理 {split_name} 集...")
            split_data = []
            
            for sample in tqdm(samples, desc=f"   {split_name}"):
                # 生成新的文件名
                new_id = sample['id']
                class_name = sample['class']
                
                # 复制RGB图像
                rgb_src = sample['rgb_source']
                rgb_dst = os.path.join(
                    self.output_folder, 'rgb', class_name, f"{new_id}.jpg"
                )
                os.makedirs(os.path.dirname(rgb_dst), exist_ok=True)
                shutil.copy(rgb_src, rgb_dst)
                
                # 处理HSI数据
                hsi_dst = os.path.join(
                    self.output_folder, 'hsi', class_name, f"{new_id}.npy"
                )
                os.makedirs(os.path.dirname(hsi_dst), exist_ok=True)
                
                if sample['hsi_source'] and os.path.exists(sample['hsi_source']):
                    # 复制真实HSI数据
                    shutil.copy(sample['hsi_source'], hsi_dst)
                elif self.create_dummy_hsi:
                    # 创建虚拟HSI数据
                    hsi_data = create_dummy_hsi(rgb_src)
                    np.save(hsi_dst, hsi_data)
                else:
                    # 创建占位符
                    hsi_data = np.zeros((224, 64, 64), dtype=np.float32)
                    np.save(hsi_dst, hsi_data)
                
                # 添加到split数据
                split_data.append({
                    'id': new_id,
                    'rgb_path': os.path.join('rgb', class_name, f"{new_id}.jpg"),
                    'hsi_path': os.path.join('hsi', class_name, f"{new_id}.npy"),
                    'class': class_name,
                    'description': sample['description']
                })
            
            # 保存split文件
            split_file = os.path.join(self.output_folder, f"{split_name}.json")
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='准备多模态病虫害数据集')
    
    parser.add_argument('--rgb_folder', type=str, required=True,
                        help='RGB图像文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='输出文件夹路径')
    parser.add_argument('--hsi_folder', type=str, default=None,
                        help='高光谱图像文件夹路径（可选）')
    parser.add_argument('--text_file', type=str, default=None,
                        help='文本描述JSON文件路径（可选）')
    parser.add_argument('--create_dummy_hsi', action='store_true',
                        help='是否创建虚拟HSI数据')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    preparer = DatasetPreparer(
        rgb_folder=args.rgb_folder,
        output_folder=args.output_folder,
        hsi_folder=args.hsi_folder,
        text_file=args.text_file,
        create_dummy_hsi=args.create_dummy_hsi,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    preparer.prepare()

if __name__ == '__main__':
    main()