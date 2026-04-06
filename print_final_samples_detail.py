"""
打印训练样本数据（最终版）
展示 Preban 和 Pick 阶段的样本结构
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

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
                })
    
    return hero_list, preban_list


# ==================== 主程序 ====================
print("="*80)
print("打印训练样本数据（最终版）")
print("="*80)

# 加载数据
with open('hero_list.json', 'r', encoding='utf-8') as f:
    hero_data = json.load(f)
    hero_list = hero_data['hero_list']
    hero_to_idx = hero_data['hero_to_idx']

idx_to_hero = {i: h for i, h in enumerate(hero_list)}

with open('output/all_complete_fast.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 找一场完整的战斗
battles = []
for server in data.get('servers', []):
    for player in server.get('players', []):
        for battle in player.get('battles', []):
            my_deck = battle.get('my_deck', {})
            enemy_deck = battle.get('enemy_deck', {})
            
            if my_deck and enemy_deck:
                my_heroes_data, _ = parse_deck(my_deck)
                enemy_heroes_data, _ = parse_deck(enemy_deck)
                
                if len(my_heroes_data) == 5 and len(enemy_heroes_data) == 5:
                    battles.append(battle)
                    if len(battles) >= 1:
                        break
        if len(battles) >= 1:
            break
    if len(battles) >= 1:
        break

battle = battles[0]
is_win = battle.get('iswin') == 1

# 解析数据
my_heroes_data, my_preban = parse_deck(battle.get('my_deck', {}))
enemy_heroes_data, enemy_preban = parse_deck(battle.get('enemy_deck', {}))

my_heroes = [h['code'] for h in my_heroes_data if h['code'] in hero_to_idx]
enemy_heroes = [h['code'] for h in enemy_heroes_data if h['code'] in hero_to_idx]
my_preban = [h for h in my_preban if h in hero_to_idx]
enemy_preban = [h for h in enemy_preban if h in hero_to_idx]

print("\n" + "="*80)
print("原始战斗数据")
print("="*80)

print(f"\n我方 Preban: {my_preban}")
print(f"敌方 Preban: {enemy_preban}")

print(f"\n我方阵容:")
for h in my_heroes_data:
    first_pick_str = " (先选)" if h.get('first_pick', 0) == 1 else ""
    print(f"  {h['code']}{first_pick_str}")

print(f"\n敌方阵容:")
for h in enemy_heroes_data:
    first_pick_str = " (先选)" if h.get('first_pick', 0) == 1 else ""
    print(f"  {h['code']}{first_pick_str}")

print(f"\n胜负：{'我方胜' if is_win else '敌方胜'}")

# 判断选秀顺序
my_first_pick = any(h.get('first_pick', 0) == 1 for h in my_heroes_data)
first_side = 'my' if my_first_pick else 'enemy'

print(f"\n先选方：{'我方' if my_first_pick else '敌方'}")

# ==================== 生成样本 ====================
print("\n" + "="*80)
print("生成训练样本")
print("="*80)

samples = []

# 1. Preban 阶段生成样本（双方各 Ban 2 个）
# Preban 特点：
# - 双方独立 Ban，不知道对面 Ban 了什么
# - 可以重复 Ban 同一个英雄（比如都 Ban c1153）
# - 所以样本是独立的，不共享序列

print("\n【Preban 阶段】（独立序列，可以重复）")
print("-"*80)

# 我方 Ban 样本（只看我方已 Ban 的）
print("\n我方 Ban（只看我方已 Ban 的）:")
my_ban_seq = []
my_ban_side = []
for i in range(len(my_preban)):
    # 生成样本：预测第 i+1 个 Ban
    sample = {
        'hero_seq': my_ban_seq.copy(),
        'side_seq': my_ban_side.copy(),
        'target': hero_to_idx[my_preban[i]],
        'phase': 0,  # Preban 阶段
        'win': 1 if is_win else 0,
        'side': 1  # 我方
    }
    samples.append(sample)
    
    hero_codes = [idx_to_hero.get(h, '?') for h in my_ban_seq]
    target_code = idx_to_hero.get(hero_to_idx[my_preban[i]], '?')
    print(f"  样本{len(samples)}: 输入={hero_codes} → 预测 Ban: {target_code}")
    
    # 添加到序列（只用于下一个我方 Ban 样本）
    my_ban_seq.append(hero_to_idx[my_preban[i]])
    my_ban_side.append(3)

# 敌方 Ban 样本（只看敌方已 Ban 的）
print("\n敌方 Ban（只看敌方已 Ban 的）:")
enemy_ban_seq = []
enemy_ban_side = []
for i in range(len(enemy_preban)):
    # 生成样本：预测第 i+1 个 Ban
    sample = {
        'hero_seq': enemy_ban_seq.copy(),
        'side_seq': enemy_ban_side.copy(),
        'target': hero_to_idx[enemy_preban[i]],
        'phase': 0,  # Preban 阶段
        'win': 0 if is_win else 1,
        'side': 2  # 敌方
    }
    samples.append(sample)
    
    hero_codes = [idx_to_hero.get(h, '?') for h in enemy_ban_seq]
    target_code = idx_to_hero.get(hero_to_idx[enemy_preban[i]], '?')
    print(f"  样本{len(samples)}: 输入={hero_codes} → 预测 Ban: {target_code}")
    
    # 添加到序列（只用于下一个敌方 Ban 样本）
    enemy_ban_seq.append(hero_to_idx[enemy_preban[i]])
    enemy_ban_side.append(3)

# 2. Pick 阶段：把 Preban 结果合并到主序列
# （Pick 阶段可以看到所有 Ban）
hero_seq = []
side_seq = []
for h in my_preban + enemy_preban:
    hero_seq.append(hero_to_idx[h])
    side_seq.append(3)

# 选秀顺序
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

print("\n【Pick 阶段】（共享序列，不能重复）")
print("-"*80)

my_idx = enemy_idx = current_phase = 0

for side, count in picks_order:
    side_name = '我方' if side == 'my' else '敌方'
    
    for _ in range(count):
        if side == 'my' and my_idx < len(my_heroes):
            if len(hero_seq) > 0:
                sample = {
                    'hero_seq': hero_seq.copy(),
                    'side_seq': side_seq.copy(),
                    'target': hero_to_idx[my_heroes[my_idx]],
                    'phase': current_phase,
                    'win': 1 if is_win else 0,
                    'side': 1
                }
                samples.append(sample)
                
                hero_codes = [idx_to_hero.get(h, '?') for h in hero_seq]
                target_code = idx_to_hero.get(hero_to_idx[my_heroes[my_idx]], '?')
                print(f"  样本{len(samples)}: 输入={hero_codes} → 预测选：{target_code} ({side_name})")
            
            hero_seq.append(hero_to_idx[my_heroes[my_idx]])
            side_seq.append(1)
            my_idx += 1
            
        elif side == 'enemy' and enemy_idx < len(enemy_heroes):
            if len(hero_seq) > 0:
                sample = {
                    'hero_seq': hero_seq.copy(),
                    'side_seq': side_seq.copy(),
                    'target': hero_to_idx[enemy_heroes[enemy_idx]],
                    'phase': current_phase,
                    'win': 0 if is_win else 1,
                    'side': 2
                }
                samples.append(sample)
                
                hero_codes = [idx_to_hero.get(h, '?') for h in hero_seq]
                target_code = idx_to_hero.get(hero_to_idx[enemy_heroes[enemy_idx]], '?')
                print(f"  样本{len(samples)}: 输入={hero_codes} → 预测选：{target_code} ({side_name})")
            
            hero_seq.append(hero_to_idx[enemy_heroes[enemy_idx]])
            side_seq.append(2)
            enemy_idx += 1
    
    if current_phase < 5:
        current_phase += 1

# ==================== 总结 ====================
print("\n" + "="*80)
print("总结")
print("="*80)

print(f"\n一场战斗生成 {len(samples)} 个样本")

print(f"\n样本分布:")
print(f"  Preban 阶段：4 个样本（独立序列）")
print(f"  Pick 阶段：10 个样本（共享序列）")

print(f"\n关键区别:")
print(f"  Preban: 双方独立，可以重复 Ban 同一个英雄")
print(f"  Pick:   共享序列，不能重复选同一个英雄")

print("\n" + "="*80)
print("样本详情（前 6 个）")
print("="*80)

for i in range(min(6, len(samples))):
    sample = samples[i]
    phase_name = "Preban" if sample['phase'] == 0 else f"Pick{sample['phase']}"
    side_name = "我方" if sample['side'] == 1 else "敌方"
    
    hero_codes = [idx_to_hero.get(h, '?') for h in sample['hero_seq']]
    target_code = idx_to_hero.get(sample['target'], '?')
    
    print(f"\n样本{i+1}:")
    print(f"  阶段：{phase_name}")
    print(f"  视角：{side_name}")
    print(f"  输入序列：{hero_codes}")
    print(f"  目标：{target_code}")
    print(f"  输入 ID: {sample['hero_seq']}")
    print(f"  目标 ID: {sample['target']}")

print("\n" + "="*80)
