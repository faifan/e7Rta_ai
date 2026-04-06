"""
目标检测示例：YOLO 简化版

目标检测 = 定位（框在哪里）+ 分类（框里是什么）

输出：[x, y, w, h, class]
- x, y: 框中心坐标
- w, h: 框宽高
- class: 类别概率
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

# ==================== 1. 创建模型 ====================
class SimpleYOLO(nn.Module):
    """
    简化版 YOLO
    
    输入：图片 [3, 224, 224]
    输出：[x, y, w, h, class1, class2, ...]
    """
    def __init__(self, num_classes=3):
        super().__init__()
        
        # CNN 提取特征
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 检测头：预测框 + 类别
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4 + num_classes)  # 4 个框参数 + 类别概率
        )
    
    def forward(self, x):
        features = self.cnn(x)
        output = self.head(features)
        return output


# ==================== 2. 准备数据 ====================
# 模拟数据：图片 + 标注
# 标注格式：[x, y, w, h, class]
# x, y: 中心坐标（0-1 归一化）
# w, h: 宽高（0-1 归一化）
# class: 类别（0=猫，1=狗，2=鸟）

train_data = [
    (torch.randn(3, 224, 224), torch.tensor([0.5, 0.5, 0.3, 0.3, 0])),  # 猫在中间
    (torch.randn(3, 224, 224), torch.tensor([0.3, 0.4, 0.2, 0.2, 1])),  # 狗在左边
    (torch.randn(3, 224, 224), torch.tensor([0.7, 0.6, 0.25, 0.25, 2])), # 鸟在右边
]


# ==================== 3. 创建模型和损失函数 ====================
num_classes = 3
model = SimpleYOLO(num_classes=num_classes)

# ⭐ 两个损失函数
box_criterion = nn.MSELoss()           # 框的位置（回归）
cls_criterion = nn.CrossEntropyLoss()  # 类别（分类）

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ==================== 4. 训练 ====================
print("="*60)
print("目标检测：YOLO 简化版")
print("="*60)

for epoch in range(50):
    total_loss = 0
    
    for images, labels in train_data:
        images = images.unsqueeze(0)  # [1, 3, 224, 224]
        labels = labels.unsqueeze(0)  # [1, 5]
        
        # 前向传播
        predictions = model(images)  # [1, 4+3]
        
        # ⭐ 分离框和类别
        pred_box = predictions[:, :4]      # [1, 4]
        pred_cls = predictions[:, 4:]      # [1, 3]
        
        true_box = labels[:, :4]           # [1, 4]
        true_cls = labels[:, 4].long()     # [1]
        
        # ⭐ 计算两个损失
        box_loss = box_criterion(pred_box, true_box)
        cls_loss = cls_criterion(pred_cls, true_cls)
        
        # 总损失
        loss = box_loss + cls_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss = {total_loss/len(train_data):.4f}")


# ==================== 5. 预测 ====================
print("\n" + "="*60)
print("预测结果")
print("="*60)

test_image = torch.randn(1, 3, 224, 224)
prediction = model(test_image)

pred_box = prediction[:, :4].detach().squeeze()
pred_cls = prediction[:, 4:].detach().squeeze()

print(f"预测框：{pred_box.tolist()}")
print(f"预测类别：{pred_cls.argmax().item()} (0=猫，1=狗，2=鸟)")


# ==================== 总结 ====================
print("\n" + "="*60)
print("目标检测要点")
print("="*60)
print("""
1. 输出 = 定位（回归）+ 分类
2. 定位用 MSELoss，分类用 CrossEntropyLoss
3. 需要标注框的位置和类别
4. YOLO: 实时检测
5. Faster R-CNN: 更准确但慢
""")
