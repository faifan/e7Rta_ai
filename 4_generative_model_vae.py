"""
生成模型示例：VAE（变分自编码器）

生成模型 = 学习数据的分布，生成新数据

VAE 原理：
1. 编码器：图片 → 潜在向量（均值 + 方差）
2. 采样：从分布中采样
3. 解码器：潜在向量 → 重建图片

损失 = 重建损失 + KL 散度
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 1. 创建模型 ====================
class VAE(nn.Module):
    """
    变分自编码器
    
    编码器：图片 → (mean, log_var)
    解码器：latent → 图片
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224→112
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112→56
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
        )
        
        # 潜在空间（均值和方差）
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 56 * 56),
            nn.ReLU(),
            nn.Unflatten(1, (64, 56, 56)),
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 56→112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2),  # 112→224
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),  # 输出 0-1
        )
    
    def encode(self, x):
        features = self.encoder(x)
        mean = self.fc_mean(features)
        log_var = self.fc_log_var(features)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        # ⭐ 重参数化技巧
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 随机噪声
        return mean + eps * std  # 采样
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var


# ==================== 2. 损失函数 ====================
def vae_loss(reconstructed, original, mean, log_var):
    """
    VAE 损失 = 重建损失 + KL 散度
    
    重建损失：图片相似度（MSE）
    KL 散度：让潜在分布接近标准正态分布
    """
    # 重建损失
    recon_loss = F.mse_loss(reconstructed, original)
    
    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    kl_loss /= mean.size(0)  # 平均到每个样本
    
    return recon_loss + kl_loss


# ==================== 3. 准备数据 ====================
# 模拟数据：一些图片
train_images = torch.randn(10, 3, 224, 224)  # 10 张图片


# ==================== 4. 创建模型和优化器 ====================
model = VAE(latent_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ==================== 5. 训练 ====================
print("="*60)
print("生成模型：VAE（变分自编码器）")
print("="*60)

for epoch in range(20):
    # 前向传播
    reconstructed, mean, log_var = model(train_images)
    
    # ⭐ VAE 损失
    loss = vae_loss(reconstructed, train_images, mean, log_var)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
        print(f"  重建损失：{F.mse_loss(reconstructed, train_images).item():.4f}")


# ==================== 6. 生成新数据 ====================
print("\n" + "="*60)
print("生成新图片")
print("="*60)

# ⭐ 从潜在空间随机采样，生成新图片
with torch.no_grad():
    # 随机采样潜在向量
    z = torch.randn(4, 16)  # 4 个随机向量
    
    # 解码生成图片
    generated = model.decode(z)
    
    print(f"输入：随机噪声 {z.shape}")
    print(f"输出：生成的图片 {generated.shape}")
    print(f"像素值范围：[{generated.min():.3f}, {generated.max():.3f}]")


# ==================== 总结 ====================
print("\n" + "="*60)
print("生成模型要点")
print("="*60)
print("""
1. VAE：变分自编码器
   - 编码器：图片 → 潜在向量
   - 解码器：潜在向量 → 图片
   - 损失：重建 + KL 散度

2. GAN：生成对抗网络
   - 生成器：噪声 → 图片
   - 判别器：判断真假
   - 对抗训练

3. 应用：
   - 生成新数据
   - 数据增强
   - 图像修复
   - 风格迁移
""")
