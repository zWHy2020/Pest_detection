# data/dataset.py
"""
多模态病虫害数据集
支持RGB图像、高光谱图像和文本描述的加载
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PestDataset(Dataset):
    """
    多模态病虫害数据集
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        rgb_transform = None,
        hsi_transform = None,
        text_model_name: str = 'bert-base-chinese',
        max_text_length: int = 128,
        use_augmentation: bool = True
    ):
        """
        Args:
            data_root: 数据集根目录
            split: 'train', 'val', 或 'test'
            rgb_transform: RGB图像变换
            hsi_transform: 高光谱图像变换
            text_model_name: 文本编码器模型名称
            max_text_length: 最大文本长度
            use_augmentation: 是否使用数据增强
        """
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.max_text_length = max_text_length
        
        # 加载数据索引
        self.samples = self._load_samples()
        
        # 加载类别映射
        with open(os.path.join(data_root, 'class_mapping.json'), 'r', encoding='utf-8') as f:
            self.class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.num_classes = len(self.class_to_idx)
        
        # 文本tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # 设置RGB图像变换
        if rgb_transform is None:
            self.rgb_transform = self._get_default_rgb_transform(use_augmentation)
        else:
            self.rgb_transform = rgb_transform
        
        # 设置HSI图像变换
        if hsi_transform is None:
            self.hsi_transform = self._get_default_hsi_transform(use_augmentation)
        else:
            self.hsi_transform = hsi_transform
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本列表"""
        split_file = os.path.join(self.data_root, f'{self.split}.json')
        with open(split_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        return samples
    
    def _get_default_rgb_transform(self, use_augmentation: bool):
        """获取默认的RGB图像变换"""
        if use_augmentation and self.split == 'train':
            transform = A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.GaussNoise(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        return transform
    
    def _get_default_hsi_transform(self, use_augmentation: bool):
        """获取默认的高光谱图像变换"""
        # 高光谱图像的数据增强需要更谨慎
        # 因为光谱信息对物理意义敏感
        return None  # 暂时返回None，后续可以添加特定的HSI增强
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            包含rgb_image, hsi_image, text, label的字典
        """
        sample = self.samples[idx]
        
        # 加载RGB图像
        rgb_path = os.path.join(self.data_root, sample['rgb_path'])
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_image = np.array(rgb_image)
        
        if self.rgb_transform:
            rgb_image = self.rgb_transform(image=rgb_image)['image']
        else:
            rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        
        # 加载高光谱图像
        hsi_path = os.path.join(self.data_root, sample['hsi_path'])
        hsi_image = np.load(hsi_path)  # 假设保存为.npy格式
        
        # HSI归一化
        hsi_image = (hsi_image - hsi_image.min()) / (hsi_image.max() - hsi_image.min() + 1e-8)
        hsi_image = torch.from_numpy(hsi_image).float()
        
        # 如果HSI是 [H, W, C] 格式，转换为 [C, H, W]
        if hsi_image.dim() == 3 and hsi_image.shape[-1] > hsi_image.shape[0]:
            hsi_image = hsi_image.permute(2, 0, 1)
        
        # 加载文本描述
        text = sample['description']
        
        # Tokenize文本
        text_encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        # 获取标签
        label = self.class_to_idx[sample['class']]
        
        return {
            'rgb_image': rgb_image,
            'hsi_image': hsi_image,
            'text_input_ids': text_encoded['input_ids'].squeeze(0),
            'text_attention_mask': text_encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'class_name': sample['class'],
            'sample_id': sample.get('id', idx)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，处理不同大小的HSI图像
    """
    # 收集所有字段
    rgb_images = torch.stack([item['rgb_image'] for item in batch])
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # HSI图像可能需要特殊处理
    hsi_images = []
    for item in batch:
        hsi = item['hsi_image']
        hsi_images.append(hsi)
    
    # 如果所有HSI图像大小相同，可以stack
    try:
        hsi_images = torch.stack(hsi_images)
    except:
        # 否则需要padding或resize到统一大小
        # 这里简单起见，假设已经是统一大小
        pass
    
    return {
        'rgb_images': rgb_images,
        'hsi_images': hsi_images,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'labels': labels,
        'class_names': [item['class_name'] for item in batch],
        'sample_ids': [item['sample_id'] for item in batch]
    }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    text_model_name: str = 'bert-base-chinese',
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_root: 数据集根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        text_model_name: 文本模型名称
        use_augmentation: 是否使用数据增强
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = PestDataset(
        data_root=data_root,
        split='train',
        text_model_name=text_model_name,
        use_augmentation=use_augmentation
    )
    
    val_dataset = PestDataset(
        data_root=data_root,
        split='val',
        text_model_name=text_model_name,
        use_augmentation=False
    )
    
    test_dataset = PestDataset(
        data_root=data_root,
        split='test',
        text_model_name=text_model_name,
        use_augmentation=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# 数据集准备工具
class DatasetBuilder:
    """
    辅助工具：从原始数据构建标准格式的数据集
    """
    @staticmethod
    def build_from_folders(
        rgb_folder: str,
        hsi_folder: str,
        text_file: str,
        output_folder: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        从文件夹构建数据集
        
        数据组织格式：
        rgb_folder/
            class1/
                img1.jpg
                img2.jpg
            class2/
                ...
        hsi_folder/
            class1/
                img1.npy
                img2.npy
            class2/
                ...
        text_file: JSON文件，格式为 {filename: description}
        """
        import shutil
        from sklearn.model_selection import train_test_split
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 加载文本描述
        with open(text_file, 'r', encoding='utf-8') as f:
            text_descriptions = json.load(f)
        
        # 收集所有样本
        samples = []
        class_names = sorted(os.listdir(rgb_folder))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for class_name in class_names:
            rgb_class_folder = os.path.join(rgb_folder, class_name)
            hsi_class_folder = os.path.join(hsi_folder, class_name)
            
            rgb_files = sorted(os.listdir(rgb_class_folder))
            
            for rgb_file in rgb_files:
                # 获取对应的HSI文件
                base_name = os.path.splitext(rgb_file)[0]
                hsi_file = base_name + '.npy'
                
                # 检查文件是否存在
                rgb_path = os.path.join(rgb_class_folder, rgb_file)
                hsi_path = os.path.join(hsi_class_folder, hsi_file)
                
                if not os.path.exists(hsi_path):
                    print(f"警告: HSI文件不存在 {hsi_path}")
                    continue
                
                # 获取文本描述
                description = text_descriptions.get(
                    base_name,
                    f"{class_name}病虫害"  # 默认描述
                )
                
                samples.append({
                    'id': base_name,
                    'rgb_path': os.path.join(class_name, rgb_file),
                    'hsi_path': os.path.join(class_name, hsi_file),
                    'description': description,
                    'class': class_name
                })
        
        # 划分数据集
        train_samples, temp_samples = train_test_split(
            samples, train_size=train_ratio, stratify=[s['class'] for s in samples],
            random_state=42
        )
        
        val_samples, test_samples = train_test_split(
            temp_samples,
            train_size=val_ratio/(val_ratio + test_ratio),
            stratify=[s['class'] for s in temp_samples],
            random_state=42
        )
        
        # 保存分割信息
        with open(os.path.join(output_folder, 'train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(output_folder, 'val.json'), 'w', encoding='utf-8') as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(output_folder, 'test.json'), 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        
        # 保存类别映射
        with open(os.path.join(output_folder, 'class_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
        
        # 复制RGB和HSI文件到输出文件夹
        for folder_name in ['rgb', 'hsi']:
            src_folder = rgb_folder if folder_name == 'rgb' else hsi_folder
            dst_folder = os.path.join(output_folder, folder_name)
            
            if os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)
            shutil.copytree(src_folder, dst_folder)
        
        print(f"数据集构建完成！")
        print(f"训练集: {len(train_samples)} 样本")
        print(f"验证集: {len(val_samples)} 样本")
        print(f"测试集: {len(test_samples)} 样本")
        print(f"类别数: {len(class_to_idx)}")


# 测试代码
if __name__ == "__main__":
    # 示例：构建数据集
    # DatasetBuilder.build_from_folders(
    #     rgb_folder='./raw_data/rgb',
    #     hsi_folder='./raw_data/hsi',
    #     text_file='./raw_data/descriptions.json',
    #     output_folder='./processed_data'
    # )
    
    # 测试数据加载
    print("数据集模块准备就绪！")
    print("请使用 DatasetBuilder.build_from_folders() 构建数据集")