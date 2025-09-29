# scripts/inference.py
"""
模型推理脚本
对单张或多张图像进行病虫害识别
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
from PIL import Image
import numpy as np
import json
from typing import List, Dict
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.main_model import MultiModalPestDetection
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PestDetector:
    """病虫害检测器"""
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"加载模型: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'args' in checkpoint:
            self.config = checkpoint['args']
        else:
            print("警告: 使用默认配置")
            self.config = {}
        
        # 加载类别映射
        data_root = self.config.get('data_root', './data')
        class_mapping_path = os.path.join(data_root, 'class_mapping.json')
        
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                self.class_to_idx = json.load(f)
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        else:
            print("警告: 找不到类别映射文件")
            self.idx_to_class = {i: f'Class_{i}' for i in range(50)}
        
        self.num_classes = len(self.idx_to_class)
        
        # 创建模型
        print("创建模型...")
        self.model = MultiModalPestDetection(
            num_classes=self.num_classes,
            rgb_image_size=self.config.get('rgb_size', 224),
            hsi_image_size=self.config.get('hsi_size', 64),
            hsi_channels=self.config.get('hsi_channels', 224),
            text_model_name=self.config.get('text_model_name', 'bert-base-chinese'),
            embed_dim=self.config.get('embed_dim', 768),
            num_heads=self.config.get('num_heads', 12),
            dropout=0.0,
            fusion_layers=self.config.get('fusion_layers', 4),
            fusion_strategy=self.config.get('fusion_strategy', 'hierarchical'),
            llm_model_name=None,
            use_lora=False
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 文本tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('text_model_name', 'bert-base-chinese')
        )
        
        # RGB图像变换
        self.rgb_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print("模型加载完成！\n")
    
    def preprocess_rgb(self, image_path: str) -> torch.Tensor:
        """预处理RGB图像"""
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        transformed = self.rgb_transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def preprocess_hsi(self, hsi_path: str) -> torch.Tensor:
        """预处理高光谱图像"""
        if hsi_path and os.path.exists(hsi_path):
            hsi = np.load(hsi_path)
            hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)
            hsi = torch.from_numpy(hsi).float()
            
            if hsi.dim() == 3 and hsi.shape[-1] > hsi.shape[0]:
                hsi = hsi.permute(2, 0, 1)
            
            return hsi.unsqueeze(0)
        else:
            return torch.zeros(1, 224, 8, 64, 64)
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """预处理文本"""
        if not text:
            text = "病虫害检测"
        
        encoded = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=128, return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    @torch.no_grad()
    def predict(
        self, rgb_path: str, hsi_path: str = None,
        text: str = None, top_k: int = 5
    ) -> Dict:
        """对单个样本进行预测"""
        rgb = self.preprocess_rgb(rgb_path).to(self.device)
        hsi = self.preprocess_hsi(hsi_path).to(self.device)
        text_encoded = self.preprocess_text(text)
        text_ids = text_encoded['input_ids'].to(self.device)
        text_mask = text_encoded['attention_mask'].to(self.device)
        
        outputs = self.model(rgb, hsi, text_ids, text_mask)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=1)
        
        top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes), dim=1)
        
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                'class_id': int(idx),
                'class_name': self.idx_to_class[int(idx)],
                'confidence': float(prob)
            })
        
        result = {
            'rgb_path': rgb_path,
            'hsi_path': hsi_path,
            'text_description': text,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None
        }
        
        return result
    
    def predict_batch(
        self, image_dir: str, output_file: str = 'predictions.json',
        pattern: str = '*.jpg', text: str = None
    ):
        """批量预测目录中的图像"""
        image_paths = glob.glob(os.path.join(image_dir, pattern))
        print(f"找到 {len(image_paths)} 张图像")
        
        results = []
        
        for image_path in image_paths:
            print(f"处理: {os.path.basename(image_path)}")
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            hsi_path = os.path.join(image_dir, base_name + '.npy')
            if not os.path.exists(hsi_path):
                hsi_path = None
            
            result = self.predict(image_path, hsi_path, text)
            results.append(result)
            
            top_pred = result['top_prediction']
            print(f"  预测: {top_pred['class_name']} "
                  f"(置信度: {top_pred['confidence']:.4f})\n")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"所有结果已保存到: {output_file}")
        return results


def parse_args():
    parser = argparse.ArgumentParser(description='病虫害识别推理')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='推理设备')
    
    parser.add_argument('--rgb_image', type=str, default=None,
                        help='RGB图像路径')
    parser.add_argument('--hsi_image', type=str, default=None,
                        help='高光谱图像路径')
    parser.add_argument('--text', type=str, default=None,
                        help='文本描述')
    
    parser.add_argument('--image_dir', type=str, default=None,
                        help='图像目录（批量推理）')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                        help='图像文件匹配模式')
    
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='输出文件路径')
    parser.add_argument('--top_k', type=int, default=5,
                        help='返回top-k预测结果')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    detector = PestDetector(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    if args.rgb_image:
        print("="*50)
        print("单张图像推理")
        print("="*50)
        
        result = detector.predict(
            rgb_path=args.rgb_image,
            hsi_path=args.hsi_image,
            text=args.text,
            top_k=args.top_k
        )
        
        print(f"\n预测结果:")
        print(f"图像: {args.rgb_image}")
        print(f"\nTop-{args.top_k} 预测:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['class_name']}: {pred['confidence']:.4f}")
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")
    
    elif args.image_dir:
        print("="*50)
        print("批量图像推理")
        print("="*50)
        
        results = detector.predict_batch(
            image_dir=args.image_dir,
            output_file=args.output,
            pattern=args.pattern,
            text=args.text
        )
    
    else:
        print("请指定 --rgb_image 或 --image_dir 参数")


if __name__ == '__main__':
    main()