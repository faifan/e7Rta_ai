"""
自监督学习示例：无需标签的学习

自监督学习 = 自己构造监督信号

核心思想：
1. 对数据进行变换（裁剪、旋转、遮挡）
2. 让模型预测变换前的样子
3. 或者让模型判断两个视图是否来自同一张图

常见方法：
1. 对比学习（SimCLR、MoCo）
2. 掩码预测（MAE、BERT）
3. 旋转预测
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 方法 1：对比学习（SimCLR 简化版） ====================
class ContrastiveLearner(nn.Module):
    """
    对比学习：让同一张图的不同视图靠近，不同图的视图远离
    
    步骤：
    1. 对图片做两次随机增强 → 两个视图
    2. 编码器提取特征
    3. 对比损失：正样本靠近，负样本远离
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # 投影头（对比学习专用）
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
    
    def forward(self, x):
        features = self.encoder(x)
        embedding = self.projector(features)
        # L2 归一化
        return F.normalize(embedding, dim=-1)


def contrastive_loss(embeddings1, embeddings2, temperature=0.5):
    """
    NT-Xent 对比损失
    
    同一张图的两个视图是正样本
    不同图的视图是负样本
    """
    batch_size = embeddings1.size(0)
    
    # 拼接所有嵌入
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # [2N, D]
    
    # 计算相似度矩阵
    similarity = embeddings @ embeddings.T / temperature  # [2N, 2N]
    
    # 标签：对角线是正样本
    labels = torch.arange(batch_size, device=embeddings.device)
    labels = torch.cat([labels, labels + batch_size])
    
    # CrossEntropyLoss
    return F.cross_entropy(similarity, labels)


# ==================== 方法 2：掩码预测（MAE 简化版） ====================
class MaskedAutoencoder(nn.Module):
    """
    掩码自编码器
    
    步骤：
    1. 随机遮挡图片的一部分（如 75%）
    2. 编码器只处理可见部分
    3. 解码器重建完整图片
    4. 损失：重建部分和原图的 MSE
    """
    def __init__(self, patch_size=16, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        
        # 编码器（只处理可见 patch）
        self.encoder = nn.Sequential(
            nn.Linear(patch_size * patch_size * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # 解码器（重建完整图片）
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_size * patch_size * 3),
            nn.Sigmoid(),
        )
    
    def forward(self, x, mask):
        # x: [N, 3, H, W]
        # mask: [N, num_patches] 1=可见，0=遮挡
        
        # 1. 分 patch
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, -1, C * self.patch_size * self.patch_size)
        
        # 2. 应用 mask
        visible = patches * mask.unsqueeze(-1)
        
        # 3. 编码
        encoded = self.encoder(visible)
        
        # 4. 解码
        reconstructed = self.decoder(encoded)
        
        return reconstructed, patches


# ==================== 方法 3：旋转预测 ====================
class RotationPredictor(nn.Module):
    """
    旋转预测：自监督任务
    
    步骤：
    1. 随机旋转图片（0°, 90°, 180°, 270°）
    2. 模型预测旋转角度
    3. 损失：CrossEntropyLoss
    
    模型必须理解图片内容才能预测旋转！
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        return self.network(x)


# ==================== 训练示例 ====================
print("="*60)
print("自监督学习：三种方法")
print("="*60)

# ----- 方法 1：对比学习 -----
print("\n【方法 1：对比学习】")

model = ContrastiveLearner(embed_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟数据：对同一张图做两次增强
images1 = torch.randn(8, 3, 64, 64)  # 增强视图 1
images2 = torch.randn(8, 3, 64, 64)  # 增强视图 2

# 前向传播
emb1 = model(images1)
emb2 = model(images2)

# 对比损失
loss = contrastive_loss(emb1, emb2)
print(f"对比损失：{loss.item():.4f}")


# ----- 方法 2：掩码预测 -----
print("\n【方法 2：掩码预测（MAE）】")

model = MaskedAutoencoder(patch_size=16, embed_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟数据
images = torch.randn(4, 3, 64, 64)
# 随机 mask（50% 可见）
mask = (torch.rand(4, 16) > 0.5).float()

# 前向传播
reconstructed, original = model(images, mask)
loss = F.mse_loss(reconstructed, original)
print(f"重建损失：{loss.item():.4f}")


# ----- 方法 3：旋转预测 -----
print("\n【方法 3：旋转预测】")

model = RotationPredictor(num_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟数据：随机旋转
images = torch.randn(8, 3, 64, 64)
rotations = torch.randint(0, 4, (8,))  # 随机旋转角度（0=0°, 1=90°, 2=180°, 3=270°）

# 前向传播
predictions = model(images)
loss = F.cross_entropy(predictions, rotations)
print(f"旋转预测损失：{loss.item():.4f}")


# ==================== 总结 ====================
print("\n" + "="*60)
print("自监督学习要点")
print("="*60)
print("""
1. 对比学习（SimCLR、MoCo）
   - 同一张图的不同视图 → 靠近
   - 不同图的视图 → 远离
   - 无需标签！

2. 掩码预测（MAE、BERT）
   - 遮挡部分内容
   - 让模型重建
   - 无需标签！

3. 旋转预测
   - 随机旋转图片
   - 预测旋转角度
   - 无需标签！

4. 应用：
   - 预训练大模型
   - 数据标注成本高时
   - 利用大量无标签数据
""")
