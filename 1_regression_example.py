"""
回归问题示例：预测房价

回归 vs 分类：
- 分类：预测类别（苹果/橘子）→ CrossEntropyLoss
- 回归：预测连续值（房价/温度）→ MSELoss
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

# ==================== 1. 创建模型 ====================
class HousePriceModel(nn.Module):
    """
    房价预测模型
    
    输入：房子特征（面积、房间数、房龄）
    输出：预测价格（连续值）
    """
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 输出 1 个值（房价）
        )
    
    def forward(self, x):
        return self.network(x)


# ==================== 2. 准备数据 ====================
# 特征：[面积 (㎡), 房间数，房龄 (年)]
# 标签：房价（万）

train_data = [
    ([80, 2, 5], 150),    # 80 平 2 室 5 年 → 150 万
    ([120, 3, 2], 280),   # 120 平 3 室 2 年 → 280 万
    ([60, 1, 10], 90),    # 60 平 1 室 10 年 → 90 万
    ([200, 4, 1], 500),   # 200 平 4 室 1 年 → 500 万
    ([90, 2, 8], 160),    # 90 平 2 室 8 年 → 160 万
    ([150, 3, 3], 350),   # 150 平 3 室 3 年 → 350 万
]

# 转成 Tensor
X_train = torch.tensor([d[0] for d in train_data], dtype=torch.float32)
y_train = torch.tensor([d[1] for d in train_data], dtype=torch.float32)

# 归一化（重要！）
X_train[:, 0] /= 200  # 面积归一化
X_train[:, 1] /= 5    # 房间数归一化
X_train[:, 2] /= 20   # 房龄归一化
y_train /= 500        # 房价归一化


# ==================== 3. 创建模型和损失函数 ====================
model = HousePriceModel(input_size=3, hidden_size=64)

# ⭐ 回归问题用 MSELoss
criterion = nn.MSELoss()

# ⭐ 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ==================== 4. 训练 ====================
print("="*60)
print("回归问题：房价预测")
print("="*60)

for epoch in range(100):
    # 前向传播
    predictions = model(X_train).squeeze()  # [6]
    
    # ⭐ 计算损失（MSE）
    loss = criterion(predictions, y_train)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")


# ==================== 5. 预测 ====================
print("\n" + "="*60)
print("预测结果")
print("="*60)

test_cases = [
    [100, 2, 5],   # 100 平 2 室 5 年
    [180, 4, 2],   # 180 平 4 室 2 年
]

for features in test_cases:
    X = torch.tensor([features], dtype=torch.float32)
    X[:, 0] /= 200
    X[:, 1] /= 5
    X[:, 2] /= 20
    
    pred = model(X).item() * 500  # 反归一化
    print(f"输入：{features} → 预测房价：{pred:.1f}万")


# ==================== 总结 ====================
print("\n" + "="*60)
print("回归问题要点")
print("="*60)
print("""
1. 标签是连续值（房价、温度、分数）
2. 损失函数用 MSELoss
3. 输出层不用激活函数（直接输出值）
4. 评估指标：MAE、RMSE、R2
""")
