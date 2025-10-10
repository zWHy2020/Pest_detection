# 多模态病虫害识别评估报告

## 基本信息

- 检查点: `./outputs/baseline_working/checkpoints/best_model.pth`
- 数据集: `./data/processed_data`
- 测试样本: 3096
- 类别数: 15
- GPU数量: 2

## 总体性能

| 指标 | 值 |
|------|----|
| 准确率 | 0.7891 |
| 精确率 | 0.7463 |
| 召回率 | 0.7567 |
| F1分数 | 0.7458 |

## 各类别性能

| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |
|------|--------|--------|--------|--------|
| Pepper__bell___Bacterial_spot | 0.808 | 0.537 | 0.645 | 149 |
| Pepper__bell___healthy | 0.812 | 0.842 | 0.827 | 221 |
| Potato___Early_blight | 0.929 | 0.873 | 0.900 | 150 |
| Potato___Late_blight | 0.727 | 0.640 | 0.681 | 150 |
| Potato___healthy | 0.289 | 0.478 | 0.361 | 23 |
| Tomato_Bacterial_spot | 0.839 | 0.884 | 0.861 | 319 |
| Tomato_Early_blight | 0.539 | 0.600 | 0.568 | 150 |
| Tomato_Late_blight | 0.729 | 0.836 | 0.779 | 287 |
| Tomato_Leaf_Mold | 0.793 | 0.776 | 0.784 | 143 |
| Tomato_Septoria_leaf_spot | 0.758 | 0.741 | 0.749 | 266 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 0.783 | 0.714 | 0.747 | 252 |
| Tomato__Target_Spot | 0.648 | 0.752 | 0.696 | 210 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 0.930 | 0.859 | 0.893 | 482 |
| Tomato__Tomato_mosaic_virus | 0.671 | 0.911 | 0.773 | 56 |
| Tomato_healthy | 0.939 | 0.908 | 0.923 | 238 |

## 可视化

- [混淆矩阵](confusion_matrix.png)
- [类别指标](per_class_metrics.png)
- [t-SNE分布](features_tsne.png)
- [PCA分布](features_pca.png)
