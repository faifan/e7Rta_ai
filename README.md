# 第七史诗选秀辅助工具 - 项目说明

## 📁 目录结构

```
D:\e7Rta\
├── 核心脚本
│   ├── 1_get_data.py              # 数据提取脚本
│   ├── 5_download_heroes.py       # 下载英雄图片
│   ├── 4_train_transformer_fixed.py # 训练脚本（防过拟合版）
│   └── 3_start_web.py             # Web 服务器启动脚本
│
├── 模型文件
│   ├── model.py                   # DraftTransformer 模型定义
│   ├── transformer_inference.py   # 推理模块
│   └── draft_transformer.pth      # 训练好的模型权重
│
├── 配置文件
│   ├── hero_list.json             # 英雄列表（348 个）
│   ├── e7.json                    # 英雄详细信息
│   └── hero_images_mapping.json   # 图片映射
│
├── 数据目录
│   └── output/
│       └── all_complete_fast.json # 5 万场战斗数据
│
├── 图片目录
│   └── hero_images/               # 英雄图片（本地缓存）
│
└── 学习示例（可选）
    ├── 1_regression_example.py     # 回归问题
    ├── 2_object_detection_example.py # 目标检测
    ├── 3_segmentation_example.py   # 语义分割
    ├── 4_generative_model_vae.py   # 生成模型
    └── 5_self_supervised_learning.py # 自监督学习
```

---

## 🚀 快速开始

### 1️⃣ 提取数据（首次运行）

```bash
python 1_get_data.py
```

**输出：** `output/all_complete_fast.json`

---

### 2️⃣ 下载英雄图片（可选，推荐）

```bash
python 5_download_heroes.py
```

**输出：**
- `hero_images/` - 348 张英雄图片
- `hero_images_mapping.json` - 图片映射

**作用：** Web 界面加载更快，不需要从官网加载

---

### 3️⃣ 训练模型

```bash
python 4_train_transformer_fixed.py
```

**防过拟合参数：**
- BATCH_SIZE = 128（增大 batch 更稳定）
- EPOCHS = 20（减少轮数防止过拟合）
- LEARNING_RATE = 0.0005（降低学习率）
- DROPOUT = 0.2（加大正则化）
- WEIGHT_DECAY = 0.01（L2 正则化）
- EARLY_STOP_PATIENCE = 5（早停机制）

**输出：** `draft_transformer.pth`

**训练时间：** 约 20-40 分钟（取决于 GPU）

**完成后自动关机：** 是（30 秒延迟）
- 取消关机：`shutdown /a`
- 永久取消：编辑脚本第 566 行，前面加 `#`

**注意：** 不再需要 `draft_weights.pth` 文件！

---

### 4️⃣ 启动 Web 服务器

```bash
python 3_start_web.py
```

**访问地址：** http://localhost:8081

**功能：**
- Preban 推荐（4 个 Ban 位）
- Pick 推荐（每选一个都更新推荐）
- Finalban 推荐
- 实时显示双方阵容

---

## 📊 数据说明

### 一场战斗的数据

```json
{
  "my_deck": {
    "preban_list": ["c1153", "c1133"],
    "hero_list": [
      {"hero_code": "c1168", "first_pick": 0},
      ...
    ]
  },
  "enemy_deck": {...},
  "iswin": 1
}
```

### 一场战斗 → 10 个训练样本

```
样本 1: [4Ban] → 预测 Pick1
样本 2: [4Ban+Pick1] → 预测 Pick2
样本 3: [4Ban+Pick1+Pick2] → 预测 Pick2
...
样本 10: [4Ban+...+Pick9] → 预测 Pick5
```

**方案 B：** 每个选择生成一个样本（完全分开）

---

## 🧠 模型说明

### DraftTransformer

```python
DraftTransformer(
    num_heroes=348,    # 英雄数量
    d_model=128,       # 特征维度
    nhead=4,           # 注意力头数
    num_layers=2,      # Transformer 层数
    dropout=0.2,       # Dropout
    max_seq_len=20     # 最大序列长度
)
```

### 输入数据

```python
hero_seqs: [batch, seq_len]  # 英雄 ID 序列
side_seqs: [batch, seq_len]  # 谁选的（1=我，2=敌，3=Ban）
phases:  [batch]             # 阶段（1-5）
masks:   [batch, seq_len]    # 掩码（1=真实，0=padding）
```

### 输出

```python
logits:   [batch, num_heroes]  # 英雄概率分布
win_pred: [batch, 1]           # 胜率预测
```

---

## 📈 训练效果

### 预期准确率

| 阶段 | 训练准确率 | 验证准确率 |
|------|-----------|-----------|
| 随机猜测 | 0.29% | 0.29% |
| 初始模型 | 10-15% | 2-5% |
| 优化后 | 12-18% | 8-12% |

**注意：** 348 选 1 非常难，Top-3/Top-5 准确率更有意义

---

## 🛠️ 常见问题

### Q1: 验证准确率低怎么办？

**A:** 已经优化了防过拟合参数：
- 增大 batch_size（64→128）
- 减少训练轮数（50→20）
- 降低学习率（0.001→0.0005）
- 加大 Dropout（0.1→0.2）
- 添加权重衰减（0.01）
- 早停机制（5 轮）

### Q2: 训练太慢怎么办？

**A:** 
- 使用 GPU（CUDA）
- 增大 batch_size（128→256）
- 减少 EPOCHS（20→15）

### Q3: Web 界面不显示推荐？

**A:**
1. 检查 `draft_transformer.pth` 是否存在
2. 检查 `hero_list.json` 是否存在
3. 重启 Web 服务器

### Q4: 图片加载慢？

**A:**
1. 运行 `python 5_download_heroes.py` 下载本地图片
2. Web 界面会使用本地图片，加载更快

---

## 📚 学习示例（可选）

这些示例帮助理解不同的机器学习任务：

### 1. 回归问题
```bash
python 1_regression_example.py
```
**学习：** 预测连续值（房价），MSELoss

### 2. 目标检测
```bash
python 2_object_detection_example.py
```
**学习：** 定位 + 分类，双损失函数

### 3. 语义分割
```bash
python 3_segmentation_example.py
```
**学习：** 像素级分类，U-Net

### 4. 生成模型
```bash
python 4_generative_model_vae.py
```
**学习：** VAE，生成新数据

### 5. 自监督学习
```bash
python 5_self_supervised_learning.py
```
**学习：** 对比学习、掩码预测、旋转预测

---

## 🌙 夜间训练说明

训练脚本已配置**完成后自动关机**：

```python
# 4_train_transformer_fixed.py 第 566 行
os.system("shutdown /s /t 30")  # 30 秒后关机
```

**取消关机：**
- 临时取消：`shutdown /a`
- 永久取消：编辑脚本，第 566 行前面加 `#`

---

## 📝 文件清单

### 核心文件（必须保留）

| 文件 | 作用 |
|------|------|
| 1_get_data.py | 数据提取 |
| 5_download_heroes.py | 下载英雄图片 |
| 4_train_transformer_fixed.py | 训练脚本 |
| 3_start_web.py | Web 服务器 |
| model.py | 模型定义 |
| transformer_inference.py | 推理模块 |
| hero_list.json | 英雄列表 |
| e7.json | 英雄信息 |
| draft_transformer.pth | 模型权重 |

### 数据文件（必须保留）

| 文件/目录 | 作用 |
|----------|------|
| output/all_complete_fast.json | 5 万场战斗数据 |
| hero_images/ | 英雄图片（本地缓存） |
| hero_images_mapping.json | 图片映射 |

### 学习示例（可选）

| 文件 | 学习内容 |
|------|---------|
| 1_regression_example.py | 回归问题 |
| 2_object_detection_example.py | 目标检测 |
| 3_segmentation_example.py | 语义分割 |
| 4_generative_model_vae.py | 生成模型 |
| 5_self_supervised_learning.py | 自监督学习 |

---

## 🎯 完整流程

```
1. python 1_get_data.py         # 提取数据
2. python 5_download_heroes.py  # 下载图片（可选）
3. python 4_train_transformer_fixed.py  # 训练模型
4. python 3_start_web.py        # 启动 Web
5. 访问 http://localhost:8081
```

---

**最后更新：** 2026 年 4 月 6 日
**版本：** v2.0（防过拟合优化版）
