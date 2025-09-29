# data/data_loader.py
"""
数据加载和预处理模块
支持RGB图像、高光谱图像和文本数据的加载
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from typing import Optional, Tuple, List, Dict
import h5py
import spectral

class PestDetectionDataset(Dataset):
    """
    病虫害检测多模态数据集
    
    支持的数据格式:
    - RGB图像: .jpg, .png
    - 高光谱图像: .npy, .mat, .h5
    - 文本描述: .txt, .json
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        tokenizer_name: str = 'bert-base-chinese',
        max_text_length: int = 256,
        rgb_transform: Optional[transforms.Compose] = None,
        hsi_transform: Optional[callable] = None,
        augment: bool = False
    ):
        """
        Args:
            data_dir: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            tokenizer_name: 文本编码器名称
            max_text_length: 最大文本长度
            rgb_transform: RGB图像变换
            hsi_transform: 高光谱图像变换
            augment: 是否使用数据增强
        """
        self.data_dir = data_dir
        self.split = split
        self.max_text_length = max_text_length
        self.augment = augment
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 设置默认的RGB变换
        if rgb_transform is None:
            self.rgb_transform = self._get_default_rgb_transform()
        else:
            self.rgb_transform = rgb_transform
            
        # 设置HSI变换
        self.hsi_transform = hsi_transform or self._get_default_hsi_transform()
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        # 加载类别映射
        self.class_to_idx = self._load_class_mapping()
        
    def _get_default_rgb_transform(self):
        """默认的RGB图像变换"""
        if self.split == 'train' and self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _get_default_hsi_transform(self):
        """默认的高光谱图像变换"""
        def transform(hsi_data):
            # 标准化
            hsi_data = (hsi_data - np.mean(hsi_data)) / (np.std(hsi_data) + 1e-8)
            
            # 调整大小到固定尺寸
            if hsi_data.shape[1:] != (64, 64):
                from scipy.ndimage import zoom
                zoom_factors = (1, 64/hsi_data.shape[1], 64/hsi_data.shape[2])
                hsi_data = zoom(hsi_data, zoom_factors, order=1)
            
            return torch.FloatTensor(hsi_data)
        
        return transform
    
    def _load_data_list(self):
        """加载数据列表"""
        list_file = os.path.join(self.data_dir, f'{self.split}.json')
        
        if os.path.exists(list_file):
            with open(list_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 如果没有预定义的列表文件，自动扫描目录
            return self._scan_data_directory()
    
    def _scan_data_directory(self):
        """扫描数据目录，自动生成数据列表"""
        data_list = []
        
        rgb_dir = os.path.join(self.data_dir, self.split, 'rgb')
        hsi_dir = os.path.join(self.data_dir, self.split, 'hsi')
        text_dir = os.path.join(self.data_dir, self.split, 'text')
        
        # 获取所有RGB图像文件
        if os.path.exists(rgb_dir):
            for filename in os.listdir(rgb_dir):
                if filename.endswith(('.jpg', '.png')):
                    base_name = os.path.splitext(filename)[0]
                    
                    # 构建对应的路径
                    rgb_path = os.path.join(rgb_dir, filename)
                    hsi_path = os.path.join(hsi_dir, f'{base_name}.npy')
                    text_path = os.path.join(text_dir, f'{base_name}.txt')
                    
                    # 从文件名或目录结构推断类别
                    label = self._infer_label(filename, rgb_path)
                    
                    data_list.append({
                        'rgb': rgb_path,
                        'hsi': hsi_path,
                        'text': text_path,
                        'label': label
                    })
        
        return data_list
    
    def _infer_label(self, filename, filepath):
        """从文件名或路径推断标签"""
        # 这里需要根据实际数据集的命名规则来实现
        # 示例：假设文件名格式为 "class_name_xxxxx.jpg"
        parts = filename.split('_')
        if len(parts) > 1:
            return parts[0]
        
        # 或者从目录结构推断
        path_parts = filepath.split(os.sep)
        for part in path_parts:
            if part in self.class_to_idx:
                return part
        
        return 'unknown'
    
    def _load_class_mapping(self):
        """加载类别映射"""
        mapping_file = os.path.join(self.data_dir, 'class_mapping.json')
        
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认的病虫害类别
            return {
                "healthy": 0,
                "aphid": 1,
                "leaf_spot": 2,
                "rust": 3,
                "powdery_mildew": 4,
                "anthracnose": 5,
                "virus": 6,
                "borer": 7,
                "armyworm": 8,
                "locust": 9,
                "spider_mite": 10,
                "whitefly": 11,
                "thrips": 12,
                "leaf_miner": 13,
                "root_rot": 14,
                "wilt": 15,
                "bacterial_blight": 16,
                "soft_rot": 17,
                "blight": 18,
                "others": 19
            }
    
    def _load_rgb_image(self, path):
        """加载RGB图像"""
        if os.path.exists(path):
            image = Image.open(path).convert('RGB')
            return self.rgb_transform(image)
        else:
            # 返回默认的空图像
            return torch.zeros(3, 224, 224)
    
    def _load_hsi_image(self, path):
        """加载高光谱图像"""
        if os.path.exists(path):
            if path.endswith('.npy'):
                hsi_data = np.load(path)
            elif path.endswith('.mat'):
                import scipy.io as sio
                mat_data = sio.loadmat(path)
                # 假设数据存储在 'data' 字段中
                hsi_data = mat_data.get('data', mat_data[list(mat_data.keys())[-1]])
            elif path.endswith('.h5'):
                with h5py.File(path, 'r') as f:
                    hsi_data = f['data'][:]
            else:
                # 使用spectral库读取其他格式
                img = spectral.open_image(path)
                hsi_data = img.load()
            
            return self.hsi_transform(hsi_data)
        else:
            # 返回默认的空高光谱图像
            return torch.zeros(224, 64, 64)
    
    def _load_text_description(self, path):
        """加载文本描述"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            # 使用默认描述
            text = "农作物叶片图像，需要进行病虫害检测"
        
        # 文本编码
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        data_item = self.data_list[idx]
        
        # 加载三种模态数据
        rgb_image = self._load_rgb_image(data_item['rgb'])
        hsi_image = self._load_hsi_image(data_item['hsi'])
        text_data = self._load_text_description(data_item['text'])
        
        # 获取标签
        label = self.class_to_idx.get(data_item['label'], 19)  # 默认为'others'
        
        return {
            'rgb_image': rgb_image,
            'hsi_image': hsi_image,
            'input_ids': text_data['input_ids'],
            'attention_mask': text_data['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': idx
        }

def create_data_loaders(
    data_path: str,
    config,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    创建数据加载器
    
    Args:
        data_path: 数据集路径
        config: 配置对象
        batch_size: 批量大小
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 创建数据集
    train_dataset = PestDetectionDataset(
        data_dir=data_path,
        split='train',
        tokenizer_name=config.text_model_name,
        max_text_length=config.max_text_length,
        augment=True
    )
    
    val_dataset = PestDetectionDataset(
        data_dir=data_path,
        split='val',
        tokenizer_name=config.text_model_name,
        max_text_length=config.max_text_length,
        augment=False
    )
    
    test_dataset = PestDetectionDataset(
        data_dir=data_path,
        split='test',
        tokenizer_name=config.text_model_name,
        max_text_length=config.max_text_length,
        augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

# 数据准备工具函数
def prepare_plantvillage_dataset(
    source_dir: str,
    target_dir: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
):
    """
    准备PlantVillage数据集
    
    Args:
        source_dir: 原始PlantVillage数据集路径
        target_dir: 目标路径
        split_ratio: 训练集、验证集、测试集的比例
    """
    import shutil
    from sklearn.model_selection import train_test_split
    
    # 创建目标目录结构
    for split in ['train', 'val', 'test']:
        for modal in ['rgb', 'hsi', 'text']:
            os.makedirs(os.path.join(target_dir, split, modal), exist_ok=True)
    
    # 收集所有图像文件
    all_images = []
    all_labels = []
    
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.png')):
                    all_images.append(os.path.join(class_path, img_file))
                    all_labels.append(class_dir)
    
    # 分割数据集
    train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=split_ratio[2], stratify=all_labels
    )
    
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_val_imgs, train_val_labels, 
        test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
        stratify=train_val_labels
    )
    
    # 复制文件到相应目录
    def copy_files(images, labels, split):
        for img_path, label in zip(images, labels):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            # 复制RGB图像
            target_rgb_path = os.path.join(target_dir, split, 'rgb', f'{label}_{base_name}.jpg')
            shutil.copy(img_path, target_rgb_path)
            
            # 生成模拟的HSI数据（实际应用中应使用真实的高光谱数据）
            hsi_data = np.random.randn(224, 64, 64).astype(np.float32)
            target_hsi_path = os.path.join(target_dir, split, 'hsi', f'{label}_{base_name}.npy')
            np.save(target_hsi_path, hsi_data)
            
            # 生成文本描述
            text_description = generate_text_description(label)
            target_text_path = os.path.join(target_dir, split, 'text', f'{label}_{base_name}.txt')
            with open(target_text_path, 'w', encoding='utf-8') as f:
                f.write(text_description)
    
    # 复制到各个分割
    copy_files(train_imgs, train_labels, 'train')
    copy_files(val_imgs, val_labels, 'val')
    copy_files(test_imgs, test_labels, 'test')
    
    # 生成数据列表文件
    for split, images, labels in [
        ('train', train_imgs, train_labels),
        ('val', val_imgs, val_labels),
        ('test', test_imgs, test_labels)
    ]:
        data_list = []
        for img_path, label in zip(images, labels):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            data_list.append({
                'rgb': f'{split}/rgb/{label}_{base_name}.jpg',
                'hsi': f'{split}/hsi/{label}_{base_name}.npy',
                'text': f'{split}/text/{label}_{base_name}.txt',
                'label': label
            })
        
        with open(os.path.join(target_dir, f'{split}.json'), 'w') as f:
            json.dump(data_list, f, indent=2)
    
    print(f"数据集准备完成！")
    print(f"训练集: {len(train_imgs)} 样本")
    print(f"验证集: {len(val_imgs)} 样本")
    print(f"测试集: {len(test_imgs)} 样本")

def generate_text_description(label: str) -> str:
    """
    根据标签生成文本描述
    
    Args:
        label: 病虫害类别标签
        
    Returns:
        文本描述
    """
    descriptions = {
        "healthy": "健康的植物叶片，没有明显的病虫害症状",
        "aphid": "叶片上可见蚜虫侵害，有小型昆虫聚集，叶片卷曲变形",
        "leaf_spot": "叶片出现褐色或黑色斑点，斑点周围有黄色晕圈",
        "rust": "叶片表面有锈色粉状斑点，严重时叶片枯黄",
        "powdery_mildew": "叶片覆盖白色粉状物质，像撒了面粉",
        "anthracnose": "叶片有不规则暗褐色病斑，中心灰白色",
        "virus": "叶片出现花叶、皱缩、畸形等病毒病症状",
        "borer": "茎秆或果实有虫孔，可见螟虫危害痕迹",
        "armyworm": "叶片被大面积啃食，有粘虫危害",
        "locust": "叶片被蝗虫啃食，呈现不规则缺刻"
    }
    
    return descriptions.get(label, "植物叶片图像，需要进行病虫害检测")

# 测试代码
if __name__ == "__main__":
    # 测试数据集
    from config.model_config import MultiModalConfig
    
    config = MultiModalConfig()
    
    # 创建测试数据集
    dataset = PestDetectionDataset(
        data_dir="./data/pest_dataset",
        split="train",
        tokenizer_name=config.text_model_name,
        max_text_length=config.max_text_length
    )
    
    # 获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"RGB图像形状: {sample['rgb_image'].shape}")
        print(f"HSI图像形状: {sample['hsi_image'].shape}")
        print(f"文本输入ID形状: {sample['input_ids'].shape}")
        print(f"标签: {sample['label']}")
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path="./data/pest_dataset",
        config=config,
        batch_size=4
    )
    
    # 测试一个批次
    for batch in train_loader:
        print(f"批次RGB形状: {batch['rgb_image'].shape}")
        print(f"批次HSI形状: {batch['hsi_image'].shape}")
        print(f"批次标签: {batch['label']}")
        break