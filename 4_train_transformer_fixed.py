"""
第七史诗选秀辅助 - 4. 训练 Transformer 模型（修正版）

修正内容：
1. 使用所有比赛数据（不是只用胜利方）
2. 根据 first_pick 判断选秀顺序
3. 学习双方的选择（不只是我方）

防过拟合改进：
1. BATCH_SIZE: 64 → 128（更稳定）
2. EPOCHS: 50 → 20（防止过拟合）
3. LEARNING_RATE: 0.001 → 0.0005（更稳定）
4. DROPOUT: 0.1 → 0.2（加大正则化）
5. WEIGHT_DECAY: 0 → 0.01（L2 正则化）
6. GRADIENT_CLIP: 1.0（梯度裁剪）
7. EARLY_STOP_PATIENCE: 10 → 5（早停）
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import random
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
from model import DraftTransformer

# ==================== 配置 ====================
DATA_FILE = 'output/all_complete_fast.json'
OUTPUT_MODEL = 'draft_transformer.pth'
HERO_LIST_FILE = 'hero_list.json'
CACHE_FILE = 'output/processed_data_cache.pth'

# ⭐ 防过拟合参数（已调整）
BATCH_SIZE = 128       # 原来 64 → 增大 batch 更稳定
EPOCHS = 20            # 原来 50 → 减少轮数防止过拟合
LEARNING_RATE = 0.0005 # 原来 0.001 → 降低学习率更稳定

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.2          # 原来 0.1 → 加大 Dropout
MAX_SEQ_LEN = 20

# ⭐ 正则化
WEIGHT_DECAY = 0.01    # 新增：L2 正则化
GRADIENT_CLIP = 1.0    # 新增：梯度裁剪

GRADIENT_ACCUM_STEPS = 1  # 不需要梯度累积了
EARLY_STOP_PATIENCE = 5   # 原来 10 → 减少耐心值
WARMUP_RATIO = 0.3        # 原来 0.1 → 增加热身比例


# ==================== 数据解析 ====================
def parse_deck(deck_data):
    """
    解析阵容数据（新格式）
    
    输入：
    {
      "hero_list": [
        {"hero_code": "c2001", "first_pick": 0, "mvp": 0, "ban": 0},
        ...
      ],
      "preban_list": ["c6005", "c1153"]
    }
    """
    if not deck_data:
        return [], []
    
    hero_list = []
    preban_list = []
    
    # 解析 preban
    if isinstance(deck_data.get('preban_list'), list):
        preban_list = deck_data['preban_list']
    
    # 解析英雄
    if isinstance(deck_data.get('hero_list'), list):
        for hero in deck_data['hero_list']:
            code = hero.get('hero_code')
            if code:
                hero_list.append({
                    'code': code,
                    'first_pick': hero.get('first_pick', 0),
                    'mvp': hero.get('mvp', 0),
                    'ban': hero.get('ban', 0)
                })
    
    return hero_list, preban_list


# ==================== 早停机制 ====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_val_acc = 0
    
    def __call__(self, val_loss, val_acc):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_val_acc = val_acc
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  早停计数：{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"  早停触发：验证损失连续{self.patience}轮未改善")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_val_acc = max(val_acc, self.best_val_acc)
            self.counter = 0
        
        return val_acc > self.best_val_acc - self.min_delta


# ==================== 数据集（修正版） ====================
class DraftDataset(Dataset):
    """
    选秀数据集（修正版）
    
    修正：
    1. 使用所有比赛（不是只用胜利方）
    2. 根据 first_pick 判断选秀顺序
    3. 学习双方的选择
    """

    def __init__(self, battles, hero_to_idx):
        self.samples = []
        self.hero_to_idx = hero_to_idx
        self.num_heroes = len(hero_to_idx)

        print("处理战斗数据...")
        for i, battle in enumerate(battles):
            self._process_battle(battle)
            if (i + 1) % 10000 == 0:
                print(f"  已处理 {i+1:,} / {len(battles):,} 场...")

        print(f"生成 {len(self.samples):,} 个训练样本")

    def _process_battle(self, battle):
        """
        处理单场战斗
        
        新逻辑：
        1. 判断谁先选（first_pick=1）
        2. 构建正确的选秀顺序
        3. 为每个选择生成样本（双方都学习）
        """
        # 解析阵容
        my_heroes_data, my_preban = parse_deck(battle.get('my_deck', {}))
        enemy_heroes_data, enemy_preban = parse_deck(battle.get('enemy_deck', {}))
        is_win = battle.get('iswin') == 1
        
        # 提取英雄代码
        my_heroes = [h['code'] for h in my_heroes_data if h['code'] in self.hero_to_idx]
        enemy_heroes = [h['code'] for h in enemy_heroes_data if h['code'] in self.hero_to_idx]
        
        # 过滤 preban
        my_preban = [h for h in my_preban if h in self.hero_to_idx]
        enemy_preban = [h for h in enemy_preban if h in self.hero_to_idx]
        
        # 检查数据有效性
        if len(my_heroes) != 5 or len(enemy_heroes) != 5:
            return  # 跳过无效数据
        
        # 判断谁先选
        # first_pick=1 表示这方先选（选 1 个）
        my_first_pick = any(h.get('first_pick', 0) == 1 for h in my_heroes_data)
        
        if my_first_pick:
            # 我方先选
            # 选秀顺序：我方 1 → 敌方 2 → 我方 2 → 敌方 2 → 我方 2 → 敌方 1
            first_side = 'my'
        else:
            # 敌方先选
            # 选秀顺序：敌方 1 → 我方 2 → 敌方 2 → 我方 2 → 敌方 2 → 我方 1
            first_side = 'enemy'
        
        # 构建选秀序列
        hero_seq = []
        side_seq = []

        # 1. Preban 阶段生成样本（双方各 Ban 2 个）
        # Preban 特点：
        # - 双方独立 Ban，不知道对面 Ban 了什么
        # - 可以重复 Ban 同一个英雄（比如都 Ban c1153）
        # - 所以样本是独立的，不共享序列
        
        # 我方 Ban 样本（只看我方已 Ban 的）
        my_ban_seq = []
        my_ban_side = []
        for i in range(len(my_preban)):
            # 生成样本：预测第 i+1 个 Ban
            self.samples.append({
                'hero_seq': my_ban_seq.copy(),
                'side_seq': my_ban_side.copy(),
                'target': self.hero_to_idx[my_preban[i]],
                'phase': 0,  # Preban 阶段
                'win': 1 if is_win else 0,
                'side': 1  # 我方
            })
            # 添加到序列（只用于下一个我方 Ban 样本）
            my_ban_seq.append(self.hero_to_idx[my_preban[i]])
            my_ban_side.append(3)
        
        # 敌方 Ban 样本（只看敌方已 Ban 的）
        enemy_ban_seq = []
        enemy_ban_side = []
        for i in range(len(enemy_preban)):
            # 生成样本：预测第 i+1 个 Ban
            self.samples.append({
                'hero_seq': enemy_ban_seq.copy(),
                'side_seq': enemy_ban_side.copy(),
                'target': self.hero_to_idx[enemy_preban[i]],
                'phase': 0,  # Preban 阶段
                'win': 0 if is_win else 1,
                'side': 2  # 敌方
            })
            # 添加到序列（只用于下一个敌方 Ban 样本）
            enemy_ban_seq.append(self.hero_to_idx[enemy_preban[i]])
            enemy_ban_side.append(3)

        # 2. Pick 阶段：把 Preban 结果合并到主序列
        # （Pick 阶段可以看到所有 Ban）
        for h in my_preban + enemy_preban:
            hero_seq.append(self.hero_to_idx[h])
            side_seq.append(3)
        if first_side == 'my':
            picks_order = [
                ('my', 1),      # 我方先选 1 个
                ('enemy', 2),   # 敌方选 2 个
                ('my', 2),      # 我方选 2 个
                ('enemy', 2),   # 敌方选 2 个
                ('my', 2),      # 我方选 2 个
                ('enemy', 1),   # 敌方选 1 个
            ]
        else:
            picks_order = [
                ('enemy', 1),   # 敌方先选 1 个
                ('my', 2),      # 我方选 2 个
                ('enemy', 2),   # 敌方选 2 个
                ('my', 2),      # 我方选 2 个
                ('enemy', 2),   # 敌方选 2 个
                ('my', 1),      # 我方选 1 个
            ]

        # 3. 生成 Pick 阶段样本
        my_idx = enemy_idx = current_phase = 0

        for side, count in picks_order:
            for _ in range(count):
                if side == 'my' and my_idx < len(my_heroes):
                    # 生成样本（学习我方选择）
                    if len(hero_seq) > 0:
                        self.samples.append({
                            'hero_seq': hero_seq.copy(),
                            'side_seq': side_seq.copy(),
                            'target': self.hero_to_idx[my_heroes[my_idx]],
                            'phase': current_phase,
                            'win': 1 if is_win else 0,
                            'side': 1  # 我方
                        })

                    # 添加到序列
                    hero_seq.append(self.hero_to_idx[my_heroes[my_idx]])
                    side_seq.append(1)  # 我方
                    my_idx += 1

                elif side == 'enemy' and enemy_idx < len(enemy_heroes):
                    # 生成样本（学习敌方选择）
                    if len(hero_seq) > 0:
                        self.samples.append({
                            'hero_seq': hero_seq.copy(),
                            'side_seq': side_seq.copy(),
                            'target': self.hero_to_idx[enemy_heroes[enemy_idx]],
                            'phase': current_phase,
                            'win': 0 if is_win else 1,
                            'side': 2
                        })

                    # 添加到序列
                    hero_seq.append(self.hero_to_idx[enemy_heroes[enemy_idx]])
                    side_seq.append(2)  # 敌方
                    enemy_idx += 1

            if current_phase < 5:
                current_phase += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'hero_seq': torch.tensor(sample['hero_seq'], dtype=torch.long),
            'side_seq': torch.tensor(sample['side_seq'], dtype=torch.long),
            'target': torch.tensor(sample['target'], dtype=torch.long),
            'phase': torch.tensor(sample['phase'], dtype=torch.long),
            'win': torch.tensor(sample['win'], dtype=torch.float),
            'side': torch.tensor(sample['side'], dtype=torch.long)
        }


def collate_fn(batch):
    max_len = max(len(item['hero_seq']) for item in batch)

    hero_seqs = torch.zeros(len(batch), max_len, dtype=torch.long)
    side_seqs = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = torch.zeros(len(batch), dtype=torch.long)
    phases = torch.zeros(len(batch), dtype=torch.long)
    wins = torch.zeros(len(batch), dtype=torch.float)
    sides = torch.zeros(len(batch), dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.float)

    for i, item in enumerate(batch):
        seq_len = len(item['hero_seq'])
        hero_seqs[i, :seq_len] = item['hero_seq']
        side_seqs[i, :seq_len] = item['side_seq']
        targets[i] = item['target']
        phases[i] = item['phase']
        wins[i] = item['win']
        sides[i] = item['side']
        masks[i, :seq_len] = 1

    return {
        'hero_seqs': hero_seqs,
        'side_seqs': side_seqs,
        'targets': targets,
        'phases': phases,
        'wins': wins,
        'sides': sides,
        'masks': masks
    }


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, accum_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        hero_seqs = batch['hero_seqs'].to(device)
        side_seqs = batch['side_seqs'].to(device)
        targets = batch['targets'].to(device)
        phases = batch['phases'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()
        logits, win_pred = model(hero_seqs, side_seqs, phases, src_mask=masks)

        loss = criterion(logits, targets)
        win_loss = nn.BCELoss()(win_pred.squeeze(), batch['wins'].to(device))
        total_loss_batch = loss + 0.3 * win_loss

        total_loss_batch = total_loss_batch / accum_steps
        total_loss_batch.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += total_loss_batch.item() * accum_steps

        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{correct/total:.4f}'
        })

        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证", unit="batch"):
            hero_seqs = batch['hero_seqs'].to(device)
            side_seqs = batch['side_seqs'].to(device)
            targets = batch['targets'].to(device)
            phases = batch['phases'].to(device)
            masks = batch['masks'].to(device)

            logits, win_pred = model(hero_seqs, side_seqs, phases, src_mask=masks)

            loss = criterion(logits, targets)
            win_loss = nn.BCELoss()(win_pred.squeeze(), batch['wins'].to(device))
            total_loss_batch = loss + 0.3 * win_loss

            total_loss += total_loss_batch.item()

            pred = logits.argmax(dim=-1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

    return total_loss / len(dataloader), correct / total


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*70)
    print("第七史诗 - 4. 训练 Transformer 模型（修正版）")
    print("="*70)

    if not os.path.exists(DATA_FILE):
        print(f"\n❌ 找不到数据文件 {DATA_FILE}")
        sys.exit(1)

    # 加载数据
    if os.path.exists(CACHE_FILE):
        print(f"\n发现缓存数据：{CACHE_FILE}")
        use_cache = input("是否使用缓存数据？(y/n): ").strip().lower()
        if use_cache == 'y':
            cache_data = torch.load(CACHE_FILE)
            hero_list = cache_data['hero_list']
            hero_to_idx = cache_data['hero_to_idx']
            idx_to_hero = cache_data['idx_to_hero']
            battles = cache_data['battles']
            print(f"✅ 已加载缓存数据：{len(battles):,} 场战斗，{len(hero_list)} 个英雄")

    if 'battles' not in locals():
        print(f"\n加载数据：{DATA_FILE}")
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_heroes = set()
        battles = []

        print("收集英雄代码...")
        for server in data.get('servers', []):
            for player in server.get('players', []):
                for battle in player.get('battles', []):
                    battles.append(battle)
                    
                    # 解析新格式
                    my_heroes_data, my_preban = parse_deck(battle.get('my_deck', {}))
                    enemy_heroes_data, enemy_preban = parse_deck(battle.get('enemy_deck', {}))
                    
                    for h in my_heroes_data + enemy_heroes_data:
                        code = h.get('code')
                        if code:
                            all_heroes.add(code)

        hero_list = sorted(list(all_heroes))
        hero_to_idx = {h: i for i, h in enumerate(hero_list)}
        idx_to_hero = {i: h for i, h in enumerate(hero_list)}

        print(f"\n保存数据缓存：{CACHE_FILE}")
        torch.save({
            'hero_list': hero_list,
            'hero_to_idx': hero_to_idx,
            'idx_to_hero': idx_to_hero,
            'battles': battles
        }, CACHE_FILE)

    print(f"英雄总数：{len(hero_list)}")
    print(f"战斗总数：{len(battles):,}")

    with open(HERO_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump({'hero_list': hero_list, 'hero_to_idx': hero_to_idx}, f, ensure_ascii=False, indent=2)

    random.shuffle(battles)
    split_idx = int(len(battles) * 0.9)
    train_battles = battles[:split_idx]
    val_battles = battles[split_idx:]

    print(f"\n训练集：{len(train_battles):,} 场")
    print(f"验证集：{len(val_battles):,} 场")

    print("\n创建训练数据集...")
    train_dataset = DraftDataset(train_battles, hero_to_idx)

    print("创建验证数据集...")
    val_dataset = DraftDataset(val_battles, hero_to_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}")

    model = DraftTransformer(
        num_heroes=len(hero_list),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 2,  # 最大学习率翻倍
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=WARMUP_RATIO,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    print("\n" + "="*70)
    print("开始训练...")
    print("="*70)
    print(f"配置参数:")
    print(f"  BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LEARNING_RATE={LEARNING_RATE}")
    print(f"  DROPOUT={DROPOUT}, WEIGHT_DECAY={WEIGHT_DECAY}, GRADIENT_CLIP={GRADIENT_CLIP}")
    print(f"  早停：{EARLY_STOP_PATIENCE} 轮无提升则停止")
    print("="*70)

    best_val_acc = 0
    start_time = datetime.now()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, epoch, accum_steps=GRADIENT_ACCUM_STEPS
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\n训练结果:")
        print(f"  训练损失：{train_loss:.4f}, 训练准确率：{train_acc:.4f}")
        print(f"  验证损失：{val_loss:.4f}, 验证准确率：{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'hero_to_idx': hero_to_idx,
                'idx_to_hero': idx_to_hero,
                'config': {
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'num_heroes': len(hero_list)
                },
                'val_acc': val_acc,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, OUTPUT_MODEL)
            print(f"✅ 保存最佳模型 (验证准确率：{val_acc:.4f})")

        early_stopping(val_loss, val_acc)
        if early_stopping.early_stop:
            print(f"\n训练提前结束，最佳验证准确率：{best_val_acc:.4f}")
            break

    elapsed = datetime.now() - start_time
    print("\n" + "="*70)
    print(f"训练完成！耗时：{elapsed}")
    print(f"最佳验证准确率：{best_val_acc:.4f}")
    print(f"模型已保存：{OUTPUT_MODEL}")
    print("="*70)
    
    # ⭐ 训练完成后关机（明天起来注释掉下面这行）
    # 如果想取消关机，运行：shutdown /a
    # print("\n⏻ 30 秒后关机，如需取消请运行：shutdown /a")
    # os.system("shutdown /s /t 30")  # 30 秒后关机，有时间取消
