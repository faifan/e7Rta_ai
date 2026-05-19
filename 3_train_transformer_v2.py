"""
第七史诗选秀辅助 - 4. 训练 Transformer 模型（修复数据问题版）

修复内容：
1. iswin 字段判断：增加对 iswin=2 的处理（可能是胜利标记）
2. 增加英雄属性 Embedding（职业 + 属性）
3. 增加模型容量
4. 增加 Label Smoothing 防止过拟合
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
DATA_FILE = 'output/all_clean_v2.json'  # ⭐ 使用二次清洗后的数据
OUTPUT_MODEL = 'draft_transformer.pth'
HERO_LIST_FILE = 'hero_list_146.json'  # ⭐ 新的英雄列表
CACHE_FILE = 'output/processed_data_cache_146.pth'
HERO_INFO_FILE = 'e7.json'

# ⭐ 防过拟合参数（2026-05-12 优化版）
BATCH_SIZE = 128
EPOCHS = 50  # ⭐ 增加训练轮数
LEARNING_RATE = 0.0005  # ⭐ 降低学习率，防止震荡

D_MODEL = 128  # ⭐ 降低模型容量，防止过拟合
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.2  # ⭐ 增加 Dropout
MAX_SEQ_LEN = 20

# ⭐ 正则化
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.1

GRADIENT_ACCUM_STEPS = 1
EARLY_STOP_PATIENCE = 10  # ⭐ 增加早停耐心
WARMUP_RATIO = 0.2


# ==================== 数据解析 ====================
def load_hero_info():
    """加载英雄属性信息"""
    with open(HERO_INFO_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hero_info = {}
    for hero in data:
        code = hero.get('code')
        if code:
            hero_info[code] = {
                'job': hero.get('job_cd', 'unknown'),
                'attr': hero.get('attribute_cd', 'unknown'),
                'grade': hero.get('grade', '5')
            }
    return hero_info


def parse_deck(deck_data):
    if not deck_data:
        return [], []

    hero_list = []
    preban_list = []

    if isinstance(deck_data.get('preban_list'), list):
        preban_list = deck_data['preban_list']

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


# ==================== 数据集（修复版） ====================
class DraftDataset(Dataset):
    def __init__(self, battles, hero_to_idx, hero_info=None):
        self.samples = []
        self.hero_to_idx = hero_to_idx
        self.hero_info = hero_info or {}
        self.num_heroes = len(hero_to_idx)

        print("处理战斗数据...")
        for i, battle in enumerate(battles):
            self._process_battle(battle)
            if (i + 1) % 10000 == 0:
                print(f"  已处理 {i+1:,} / {len(battles):,} 场...")

        print(f"生成 {len(self.samples):,} 个训练样本")
        
        # 统计标签分布
        win_samples = sum(1 for s in self.samples if s['win'] == 1)
        print(f"胜率标签分布：胜 {win_samples:,} ({win_samples/len(self.samples)*100:.1f}%) / 负 {len(self.samples)-win_samples:,}")

    def _process_battle(self, battle):
        my_heroes_data, my_preban = parse_deck(battle.get('my_deck', {}))
        enemy_heroes_data, enemy_preban = parse_deck(battle.get('enemy_deck', {}))
        
        # ⭐ 修复：iswin 可能是 1 或 2 都表示胜利
        iswin_raw = battle.get('iswin', 0)
        is_win = iswin_raw in [1, 2]  # 1 或 2 都算赢

        my_heroes = [h['code'] for h in my_heroes_data if h['code'] in self.hero_to_idx]
        enemy_heroes = [h['code'] for h in enemy_heroes_data if h['code'] in self.hero_to_idx]

        my_preban = [h for h in my_preban if h in self.hero_to_idx]
        enemy_preban = [h for h in enemy_preban if h in self.hero_to_idx]

        if len(my_heroes) != 5 or len(enemy_heroes) != 5:
            return

        my_first_pick = any(h.get('first_pick', 0) == 1 for h in my_heroes_data)

        if my_first_pick:
            first_side = 'my'
        else:
            first_side = 'enemy'

        hero_seq = []
        side_seq = []

        # Preban 阶段（跳过空上文样本，空序列进 Transformer 全 masked 输出无意义）
        my_ban_seq = []
        my_ban_side = []
        for i in range(len(my_preban)):
            if len(my_ban_seq) > 0:
                self.samples.append({
                    'hero_seq': my_ban_seq.copy(),
                    'side_seq': my_ban_side.copy(),
                    'target': self.hero_to_idx[my_preban[i]],
                    'phase': 0,
                    'win': 1 if is_win else 0,
                    'side': 1
                })
            my_ban_seq.append(self.hero_to_idx[my_preban[i]])
            my_ban_side.append(3)

        enemy_ban_seq = []
        enemy_ban_side = []
        for i in range(len(enemy_preban)):
            if len(enemy_ban_seq) > 0:
                self.samples.append({
                    'hero_seq': enemy_ban_seq.copy(),
                    'side_seq': enemy_ban_side.copy(),
                    'target': self.hero_to_idx[enemy_preban[i]],
                    'phase': 0,
                    'win': 0 if is_win else 1,
                    'side': 2
                })
            enemy_ban_seq.append(self.hero_to_idx[enemy_preban[i]])
            enemy_ban_side.append(3)

        for h in my_preban + enemy_preban:
            hero_seq.append(self.hero_to_idx[h])
            side_seq.append(3)
            
        if first_side == 'my':
            picks_order = [
                ('my', 1),
                ('enemy', 2),
                ('my', 2),
                ('enemy', 2),
                ('my', 2),
                ('enemy', 1),
            ]
        else:
            picks_order = [
                ('enemy', 1),
                ('my', 2),
                ('enemy', 2),
                ('my', 2),
                ('enemy', 2),
                ('my', 1),
            ]

        my_idx = enemy_idx = current_phase = 0

        for side, count in picks_order:
            for _ in range(count):
                if side == 'my' and my_idx < len(my_heroes):
                    if len(hero_seq) > 0:
                        self.samples.append({
                            'hero_seq': hero_seq.copy(),
                            'side_seq': side_seq.copy(),
                            'target': self.hero_to_idx[my_heroes[my_idx]],
                            'phase': current_phase,
                            'win': 1 if is_win else 0,
                            'side': 1
                        })
                    hero_seq.append(self.hero_to_idx[my_heroes[my_idx]])
                    side_seq.append(1)
                    my_idx += 1

                elif side == 'enemy' and enemy_idx < len(enemy_heroes):
                    if len(hero_seq) > 0:
                        self.samples.append({
                            'hero_seq': hero_seq.copy(),
                            'side_seq': side_seq.copy(),
                            'target': self.hero_to_idx[enemy_heroes[enemy_idx]],
                            'phase': current_phase,
                            'win': 0 if is_win else 1,
                            'side': 2
                        })
                    hero_seq.append(self.hero_to_idx[enemy_heroes[enemy_idx]])
                    side_seq.append(2)
                    enemy_idx += 1

            if current_phase < 5:
                current_phase += 1

        # ==================== Finalban 阶段 ====================
        # 找出被 Finalban 的英雄（ban=1 表示被禁用）
        my_finalban = None
        enemy_finalban = None

        for h in my_heroes_data:
            if h.get('ban') == 1:
                my_finalban = h['code']
                break

        for h in enemy_heroes_data:
            if h.get('ban') == 1:
                enemy_finalban = h['code']
                break

        # 我方 Finalban：从敌方 5 英雄中选 1 个禁用
        if enemy_finalban and enemy_finalban in self.hero_to_idx:
            self.samples.append({
                'hero_seq': hero_seq.copy(),
                'side_seq': side_seq.copy(),
                'target': self.hero_to_idx[enemy_finalban],
                'phase': 6,  # Finalban 阶段
                'win': 1 if is_win else 0,
                'side': 1
            })

        # 敌方 Finalban：从我方 5 英雄中选 1 个禁用
        if my_finalban and my_finalban in self.hero_to_idx:
            self.samples.append({
                'hero_seq': hero_seq.copy(),
                'side_seq': side_seq.copy(),
                'target': self.hero_to_idx[my_finalban],
                'phase': 6,  # Finalban 阶段
                'win': 0 if is_win else 1,
                'side': 2
            })

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
    top3_correct = 0
    top5_correct = 0
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

        logits, win_pred = model(hero_seqs, side_seqs, phases, src_mask=masks)

        loss = criterion(logits, targets)
        win_loss = nn.BCELoss()(win_pred.squeeze(1), batch['wins'].to(device))
        total_loss_batch = loss + 0.1 * win_loss

        total_loss_batch = total_loss_batch / accum_steps
        total_loss_batch.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += total_loss_batch.item() * accum_steps

        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        top3_correct += (logits.topk(3, dim=-1).indices == targets.unsqueeze(1)).any(dim=1).sum().item()
        top5_correct += (logits.topk(5, dim=-1).indices == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{correct/total:.4f}',
            'top3': f'{top3_correct/total:.4f}'
        })

        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

    return total_loss / len(dataloader), correct / total, top3_correct / total, top5_correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    top3_correct = 0
    top5_correct = 0
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
            win_loss = nn.BCELoss()(win_pred.squeeze(1), batch['wins'].to(device))
            total_loss_batch = loss + 0.1 * win_loss

            total_loss += total_loss_batch.item()

            pred = logits.argmax(dim=-1)
            correct += (pred == targets).sum().item()
            top3_correct += (logits.topk(3, dim=-1).indices == targets.unsqueeze(1)).any(dim=1).sum().item()
            top5_correct += (logits.topk(5, dim=-1).indices == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

    return total_loss / len(dataloader), correct / total, top3_correct / total, top5_correct / total


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*70)
    print("第七史诗 - 4. 训练 Transformer 模型（修复数据问题版）")
    print("="*70)

    if not os.path.exists(DATA_FILE):
        print(f"\n❌ 找不到数据文件 {DATA_FILE}")
        sys.exit(1)

    # 加载英雄信息
    print("\n加载英雄属性信息...")
    hero_info = load_hero_info()
    print(f"✅ 加载 {len(hero_info)} 个英雄属性")

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
        for battle in data:
            battles.append(battle)

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
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
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

    # 去重：同一场对局从双方视角各记录一次，用 battle_seq 精确去重
    print("\n对局去重...")
    seen_seqs = set()
    deduped = []
    for battle in battles:
        key = battle.get('battle_seq')
        if key is None or key not in seen_seqs:
            if key is not None:
                seen_seqs.add(key)
            deduped.append(battle)
    print(f"去重前：{len(battles):,}  去重后：{len(deduped):,}  删除：{len(battles)-len(deduped):,}")
    battles = deduped

    random.shuffle(battles)
    split_idx = int(len(battles) * 0.9)
    train_battles = battles[:split_idx]
    val_battles = battles[split_idx:]

    print(f"\n训练集：{len(train_battles):,} 场")
    print(f"验证集：{len(val_battles):,} 场")

    print("\n创建训练数据集...")
    train_dataset = DraftDataset(train_battles, hero_to_idx, hero_info)

    print("创建验证数据集...")
    val_dataset = DraftDataset(val_battles, hero_to_idx, hero_info)

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
        max_seq_len=MAX_SEQ_LEN,
        dim_feedforward=D_MODEL * 4  # ⭐ FFN 维度应该是 d_model 的 4 倍
    ).to(device)

    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 2,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=WARMUP_RATIO,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    # ⭐ 使用 Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    print("\n" + "="*70)
    print("开始训练...")
    print("="*70)
    print(f"配置参数:")
    print(f"  BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LEARNING_RATE={LEARNING_RATE}")
    print(f"  D_MODEL={D_MODEL}, NHEAD={NHEAD}, NUM_LAYERS={NUM_LAYERS}")
    print(f"  DROPOUT={DROPOUT}, WEIGHT_DECAY={WEIGHT_DECAY}, GRADIENT_CLIP={GRADIENT_CLIP}")
    print(f"  LABEL_SMOOTHING={LABEL_SMOOTHING}")
    print(f"  DIM_FEEDFORWARD={D_MODEL * 4}")
    print(f"  早停：{EARLY_STOP_PATIENCE} 轮无提升则停止")
    print("="*70)

    best_val_acc = 0
    start_time = datetime.now()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 50)

        train_loss, train_acc, train_top3, train_top5 = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, epoch, accum_steps=GRADIENT_ACCUM_STEPS
        )
        val_loss, val_acc, val_top3, val_top5 = evaluate(model, val_loader, criterion, device)

        print(f"\n训练结果:")
        print(f"  训练损失：{train_loss:.4f}  Top1：{train_acc:.4f}  Top3：{train_top3:.4f}  Top5：{train_top5:.4f}")
        print(f"  验证损失：{val_loss:.4f}  Top1：{val_acc:.4f}  Top3：{val_top3:.4f}  Top5：{val_top5:.4f}")

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
                    'num_heroes': len(hero_list),
                    'dim_feedforward': D_MODEL * 4  # 保存 FFN 维度
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
