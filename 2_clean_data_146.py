"""
清理数据：
  第一遍：过滤无效英雄、不完整对局
  第二遍：过滤包含低频英雄（出现次数 < MIN_APPEAR）的对局
最终结果保存到 output/all_clean_v2.json
"""
import json
import os
from datetime import datetime, timedelta

MIN_APPEAR = 50
KEEP_DAYS = 15  # 只保留最近 N 天的战斗数据
CUTOFF_DATE = datetime.now() - timedelta(days=KEEP_DAYS)

# 加载 146 个英雄
with open('e7.json', 'r', encoding='utf-8') as f:
    _hero_data = json.load(f)
valid_heroes = set(h['code'] for h in _hero_data)
code_to_name = {h['code']: h['name'] for h in _hero_data}

def _n(code):
    name = code_to_name.get(code)
    return f"{name}({code})" if name else code

print(f"有效英雄数量：{len(valid_heroes)}")

# 加载原始数据
print("\n加载对战数据...")
with open('output/all_complete_fast.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ── 第一遍清洗：过滤无效数据 ─────────────────────────────────
total_battles = 0
invalid_reasons = {
    'my_hero_count': 0,
    'enemy_hero_count': 0,
    'my_invalid_hero': 0,
    'enemy_invalid_hero': 0,
    'my_preban_invalid': 0,
    'enemy_preban_invalid': 0,
    'my_preban_count': 0,
    'enemy_preban_count': 0,
    'too_old': 0,
    'no_opening_rule': 0,
}

pass1_battles = []

for server in data.get('servers', []):
    for player in server.get('players', []):
        for battle in player.get('battles', []):
            total_battles += 1

            my_deck    = battle.get('my_deck', {})
            enemy_deck = battle.get('enemy_deck', {})
            my_heroes    = my_deck.get('hero_list', [])
            enemy_heroes = enemy_deck.get('hero_list', [])
            my_preban    = my_deck.get('preban_list', [])
            enemy_preban = enemy_deck.get('preban_list', [])

            if len(my_preban) != 2:
                invalid_reasons['my_preban_count'] += 1; continue
            if len(enemy_preban) != 2:
                invalid_reasons['enemy_preban_count'] += 1; continue
            if len(my_heroes) != 5:
                invalid_reasons['my_hero_count'] += 1; continue
            if len(enemy_heroes) != 5:
                invalid_reasons['enemy_hero_count'] += 1; continue

            my_codes = [h.get('hero_code') for h in my_heroes]
            if any(c not in valid_heroes for c in my_codes):
                invalid_reasons['my_invalid_hero'] += 1; continue

            enemy_codes = [h.get('hero_code') for h in enemy_heroes]
            if any(c not in valid_heroes for c in enemy_codes):
                invalid_reasons['enemy_invalid_hero'] += 1; continue

            if any(c not in valid_heroes for c in my_preban):
                invalid_reasons['my_preban_invalid'] += 1; continue
            if any(c not in valid_heroes for c in enemy_preban):
                invalid_reasons['enemy_preban_invalid'] += 1; continue

            battle_day_str = battle.get('battle_day', '')
            try:
                battle_date = datetime.strptime(battle_day_str[:19], '%Y-%m-%d %H:%M:%S')
                if battle_date < CUTOFF_DATE:
                    invalid_reasons['too_old'] += 1; continue
            except (ValueError, TypeError):
                invalid_reasons['too_old'] += 1; continue

            opening_rule = battle.get('opening_rule_title', '_')
            if opening_rule == '_':
                invalid_reasons['no_opening_rule'] += 1; continue

            pass1_battles.append({
                'battle_seq': battle.get('battle_seq'),
                'iswin': battle.get('iswin', 0),
                'opening_rule_title': opening_rule,
                'my_deck': {
                    'preban_list': my_preban,
                    'hero_list': [
                        {'hero_code': h.get('hero_code'), 'first_pick': h.get('first_pick', 0), 'ban': h.get('ban', 0)}
                        for h in my_heroes
                    ]
                },
                'enemy_deck': {
                    'preban_list': enemy_preban,
                    'hero_list': [
                        {'hero_code': h.get('hero_code'), 'first_pick': h.get('first_pick', 0), 'ban': h.get('ban', 0)}
                        for h in enemy_heroes
                    ]
                }
            })

pass1_count = len(pass1_battles)
print(f"\n{'='*60}")
print("第一遍清洗结果")
print(f"{'='*60}")
print(f"原始对局数：{total_battles:,}")
print(f"有效对局数：{pass1_count:,} ({pass1_count/total_battles*100:.1f}%)")
print(f"无效对局数：{total_battles - pass1_count:,} ({(total_battles - pass1_count)/total_battles*100:.1f}%)")
print("\n无效原因分布:")
for reason, count in invalid_reasons.items():
    if count:
        print(f"  {reason}: {count:,} ({count/total_battles*100:.1f}%)")

# ── 第二遍清洗：过滤低频英雄 ─────────────────────────────────
hero_appear = {}
for battle in pass1_battles:
    for code in (
        [h['hero_code'] for h in battle['my_deck']['hero_list']] +
        [h['hero_code'] for h in battle['enemy_deck']['hero_list']] +
        battle['my_deck']['preban_list'] +
        battle['enemy_deck']['preban_list']
    ):
        hero_appear[code] = hero_appear.get(code, 0) + 1

low_freq = {code for code, cnt in hero_appear.items() if cnt < MIN_APPEAR}

print(f"\n{'='*60}")
print(f"出现次数 < {MIN_APPEAR} 的低频英雄（{len(low_freq)} 个）")
print(f"{'='*60}")
for code in sorted(low_freq, key=lambda c: hero_appear[c]):
    print(f"  {_n(code)}: {hero_appear[code]} 次")

pass2_battles = [
    b for b in pass1_battles
    if not (
        set([h['hero_code'] for h in b['my_deck']['hero_list']] +
            [h['hero_code'] for h in b['enemy_deck']['hero_list']] +
            b['my_deck']['preban_list'] +
            b['enemy_deck']['preban_list'])
        & low_freq
    )
]

pass2_count = len(pass2_battles)
removed = pass1_count - pass2_count
print(f"\n{'='*60}")
print("第二遍清洗结果")
print(f"{'='*60}")
print(f"输入对局数：{pass1_count:,}")
print(f"删除对局数：{removed:,} ({removed/pass1_count*100:.1f}%)")
print(f"最终对局数：{pass2_count:,} ({pass2_count/pass1_count*100:.1f}%)")

# ── 保存最终结果 ──────────────────────────────────────────────
output_file = 'output/all_clean_v2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(pass2_battles, f, ensure_ascii=False)
print(f"\n已保存到：{output_file}")
print(f"文件大小：{round(os.path.getsize(output_file) / 1024 / 1024, 2)} MB")

# ── 最终英雄出现次数 ──────────────────────────────────────────
hero_appear_final = {}
for battle in pass2_battles:
    for code in (
        [h['hero_code'] for h in battle['my_deck']['hero_list']] +
        [h['hero_code'] for h in battle['enemy_deck']['hero_list']] +
        battle['my_deck']['preban_list'] +
        battle['enemy_deck']['preban_list']
    ):
        hero_appear_final[code] = hero_appear_final.get(code, 0) + 1

print(f"\n{'='*60}")
print(f"最终英雄出现次数（共 {len(hero_appear_final)} 个英雄，按降序）")
print(f"{'='*60}")
for i, (code, cnt) in enumerate(sorted(hero_appear_final.items(), key=lambda x: -x[1])):
    print(f"  {i+1:3d}. {_n(code)}: {cnt:,} 次")
