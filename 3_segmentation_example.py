"""
语义分割示例：U-Net 简化版

语义分割 = 像素级分类
每个像素都要预测类别

输入：[3, 224, 224] 图片
输出：[num_classes, 224, 224] 每个像素的类别概率
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

# ==================== 1. 创建模型 ====================
class SimpleUNet(nn.Module):
    """
    简化版 U-Net
    
    特点：
    - 编码器：下采样提取特征
    - 解码器：上采样恢复分辨率
    - 跳跃连接：保留细节
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224→112
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112→56
        )
        
        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 56→112
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 112→224
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        
        # 分类头
        self.classifier = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        
        # 解码
        features = self.decoder(features)
        
        # 分类
        output = self.classifier(features)
        
        return output


# ==================== 2. 准备数据 ====================
# 输入：图片 [3, 224, 224]
# 输出：分割掩码 [224, 224] 每个像素一个类别
# 0=背景，1=前景

batch_size = 2
images = torch.randn(batch_size, 3, 224, 224)  # 2 张图片
masks = torch.randint(0, 2, (batch_size, 224, 224)).long()  # 2 个掩码


# ==================== 3. 创建模型和损失函数 ====================
num_classes = 2
model = SimpleUNet(num_classes=num_classes)

# ⭐ 像素级分类：CrossEntropyLoss
# 输入：[batch, num_classes, H, W]
# 目标：[batch, H, W]
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ==================== 4. 训练 ====================
print("="*60)
print("语义分割：U-Net 简化版")
print("="*60)

for epoch in range(20):
    # 前向传播
    predictions = model(images)  # [2, 2, 224, 224]
    
    # ⭐ 计算损失（像素级分类）
    loss = criterion(predictions, masks)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")


# ==================== 5. 预测 ====================
print("\n" + "="*60)
print("预测结果")
print("="*60)

test_image = torch.randn(1, 3, 224, 224)
prediction = model(test_image)  # [1, 2, 224, 224]

# 取概率最大的类别
pred_mask = prediction.argmax(dim=1)  # [1, 224, 224]

print(f"输入形状：{test_image.shape}")
print(f"输出形状：{pred_mask.shape}")
print(f"每个像素的类别：0=背景，1=前景")


# ==================== 总结 ====================
print("\n" + "="*60)
print("语义分割要点")
print("="*60)
print("""
1. 输出是掩码（每个像素一个类别）
2. 用 CrossEntropyLoss（像素级分类）
3. U-Net：编码器 - 解码器结构
4. 应用：医学图像、自动驾驶、抠图
5. DeepLab：空洞卷积，更大感受野
""")
