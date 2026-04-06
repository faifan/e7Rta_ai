"""
第七史诗选秀辅助 - 完整流程版

使用 Transformer 模型进行英雄推荐
支持退回上一步、e7.json 映射、多条件搜索
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys
from collections import defaultdict
from datetime import datetime

# 导入 Transformer 推理
from transformer_inference import DraftRecommender

sys.stdout.reconfigure(encoding='utf-8')

PORT = 8081
TRANSFORMER_MODEL = 'draft_transformer.pth'
HERO_LIST_FILE = 'hero_list.json'
E7_JSON_FILE = 'e7.json'
HERO_IMAGES_DIR = 'hero_images'  # 本地英雄图片目录
HERO_IMAGES_MAPPING = 'hero_images_mapping.json'  # 图片映射文件

# 检查是否使用本地图片
USE_LOCAL_IMAGES = os.path.exists(HERO_IMAGES_DIR) and os.path.exists(HERO_IMAGES_MAPPING)

# 加载图片映射
LOCAL_IMAGE_MAPPING = {}
if USE_LOCAL_IMAGES:
    try:
        with open(HERO_IMAGES_MAPPING, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            LOCAL_IMAGE_MAPPING = mapping_data.get('mapping', {})
            print(f"✅ 加载本地图片映射：{len(LOCAL_IMAGE_MAPPING)} 个英雄")
    except Exception as e:
        print(f"⚠️ 加载图片映射失败：{e}")
        USE_LOCAL_IMAGES = False
else:
    print("⚠️ 未找到本地图片，将使用官网图片")

def get_hero_image_url(code):
    """获取英雄图片 URL"""
    if USE_LOCAL_IMAGES and code in LOCAL_IMAGE_MAPPING:
        return f'/hero_images/{LOCAL_IMAGE_MAPPING[code]}'
    else:
        return f'https://static-pubcomm.onstove.com/event/live/epic7/guide/images/hero/{code}_s.png'

# ==================== 加载英雄列表 ====================
print("="*70)
print("第七史诗选秀辅助")
print("="*70)

print(f"\n加载英雄列表：{HERO_LIST_FILE}")
if not os.path.exists(HERO_LIST_FILE):
    print(f"\n❌ 找不到英雄列表 {HERO_LIST_FILE}")
    print("请先运行 1_get_data.py 生成英雄列表")
    sys.exit(1)

with open(HERO_LIST_FILE, 'r', encoding='utf-8') as f:
    hero_data = json.load(f)
    hero_list = hero_data['hero_list']
    hero_to_idx = hero_data['hero_to_idx']
    num_heroes = len(hero_list)

print(f"✅ 英雄列表加载成功")
print(f"  英雄数：{num_heroes}")

# 初始化 hero_stats（空字典，用于备用推荐）
hero_stats = {}

# ==================== 加载 e7.json 映射 ====================
print(f"\n加载英雄映射：{E7_JSON_FILE}")
E7_HEROES = {}  # code -> {grade, name, job_cd, attribute_cd}
HERO_BY_NAME = {}  # name -> code
HERO_BY_JOB = defaultdict(list)  # job_cd -> [codes]
HERO_BY_ATTRIBUTE = defaultdict(list)  # attribute_cd -> [codes]
HERO_BY_GRADE = defaultdict(list)  # grade -> [codes]

if os.path.exists(E7_JSON_FILE):
    with open(E7_JSON_FILE, 'r', encoding='utf-8') as f:
        e7_data = json.load(f)
        for hero in e7_data:
            code = hero.get('code', '')
            E7_HEROES[code] = {
                'grade': hero.get('grade', '0'),
                'name': hero.get('name', ''),
                'job_cd': hero.get('job_cd', ''),
                'attribute_cd': hero.get('attribute_cd', '')
            }
            HERO_BY_NAME[hero.get('name', '')] = code
            HERO_BY_JOB[hero.get('job_cd', '')].append(code)
            HERO_BY_ATTRIBUTE[hero.get('attribute_cd', '')].append(code)
            HERO_BY_GRADE[hero.get('grade', '')].append(code)
    print(f"✅ 加载 {len(E7_HEROES)} 个英雄映射")
else:
    print(f"⚠️ 找不到 {E7_JSON_FILE}，将使用基础映射")

def get_hero_info(code):
    """获取英雄完整信息"""
    if code in E7_HEROES:
        info = E7_HEROES[code].copy()
        info['code'] = code
        return info
    return {'code': code, 'grade': '0', 'name': code, 'job_cd': '', 'attribute_cd': ''}

def get_hero_name(code):
    """获取英雄名称"""
    if code in E7_HEROES:
        return E7_HEROES[code].get('name', code)
    return code

def search_heroes(query_grade='', query_name='', query_job='', query_attribute=''):
    """
    多条件联合搜索英雄
    
    Args:
        query_grade: 星级 (3,4,5)
        query_name: 名称（支持模糊）
        query_job: 职业 (knight, warrior, ranger, assassin, mage, priest)
        query_attribute: 属性 (fire, ice, wind, dark, light)
    
    Returns:
        List[str]: 匹配的英雄代码列表
    """
    results = set(hero_list)  # 从权重的英雄列表开始
    
    # 星级筛选
    if query_grade and query_grade in HERO_BY_GRADE:
        grade_codes = set(HERO_BY_GRADE[query_grade])
        results = results & grade_codes
    
    # 名称筛选（模糊匹配）
    if query_name:
        name_results = set()
        for name, code in HERO_BY_NAME.items():
            if query_name.lower() in name.lower() and code in hero_list:
                name_results.add(code)
        # 也支持 code 直接匹配
        if query_name in hero_list:
            name_results.add(query_name)
        results = results & name_results if name_results else results
    
    # 职业筛选
    if query_job and query_job in HERO_BY_JOB:
        job_codes = set(HERO_BY_JOB[query_job])
        results = results & job_codes
    
    # 属性筛选
    if query_attribute and query_attribute in HERO_BY_ATTRIBUTE:
        attr_codes = set(HERO_BY_ATTRIBUTE[query_attribute])
        results = results & attr_codes
    
    return sorted(list(results))

# ==================== 游戏状态 ====================
game_state = {
    'my_first': True, 
    'phase': 'preban', 
    'step': 0,
    'my_preban': [], 
    'enemy_preban': [], 
    'my_picks': [], 
    'enemy_picks': [],
    'my_final_ban': [], 
    'enemy_final_ban': [],
    'history': []  # 历史记录，用于退回
}

def reset_game(my_first):
    global game_state
    game_state = {
        'my_first': my_first, 
        'phase': 'preban', 
        'step': 0,
        'my_preban': [], 
        'enemy_preban': [], 
        'my_picks': [], 
        'enemy_picks': [],
        'my_final_ban': [], 
        'enemy_final_ban': [],
        'history': []
    }

def save_state():
    """保存当前状态到历史记录"""
    state_copy = {
        'my_first': game_state['my_first'],
        'phase': game_state['phase'],
        'step': game_state['step'],
        'my_preban': game_state['my_preban'].copy(),
        'enemy_preban': game_state['enemy_preban'].copy(),
        'my_picks': game_state['my_picks'].copy(),
        'enemy_picks': game_state['enemy_picks'].copy(),
        'my_final_ban': game_state['my_final_ban'].copy(),
        'enemy_final_ban': game_state['enemy_final_ban'].copy(),
    }
    game_state['history'].append(state_copy)
    # 限制历史记录长度
    if len(game_state['history']) > 20:
        game_state['history'].pop(0)

def undo_last_action():
    """退回上一步"""
    if not game_state['history']:
        return False
    
    last_state = game_state['history'].pop()
    game_state.update(last_state)
    return True

def get_banned():
    return set(game_state['my_preban'] + game_state['enemy_preban'])

def get_used():
    return set(game_state['my_picks'] + game_state['enemy_picks'])

def get_available():
    return [h for h in hero_list if h not in (get_used() | get_banned())]

def get_recommendations(count=5):
    """
    根据当前阶段推荐英雄 - 全部使用 Transformer 模型
    """
    phase = game_state['phase']

    # Preban + Pick 阶段：全部使用 Transformer 模型推荐
    if use_transformer and (phase == 'preban' or phase.startswith('pick')):
        banned = game_state['my_preban'] + game_state['enemy_preban']

        # Preban 阶段：使用模型推荐（可以重复 Ban）
        # Pick 阶段：使用模型推荐（不能重复选）
        if phase == 'preban':
            recs = recommender.recommend_preban_simple(
                my_banned=game_state['my_preban'],
                enemy_banned=game_state['enemy_preban'],
                top_k=count
            )
        else:
            recs = recommender.recommend(
                my_picks=game_state['my_picks'],
                enemy_picks=game_state['enemy_picks'],
                banned=banned,
                phase=phase,
                my_first=game_state['my_first'],
                top_k=count
            )

        result = []
        for rec in recs:
            info = get_hero_info(rec['hero_code'])
            result.append({
                'hero_code': rec['hero_code'],
                'hero_name': info.get('name', rec['hero_code']),
                'grade': info.get('grade', '0'),
                'job_cd': info.get('job_cd', ''),
                'attribute_cd': info.get('attribute_cd', ''),
                'score': round(rec['probability'] * 1000, 1),
                'win_rate': round(rec['win_rate'] * 100, 1) if rec.get('win_rate') else 50.0,
                'probability': round(rec['probability'] * 100, 2) if rec.get('probability') else 0,
                'picks': 0,
                'model': 'transformer'
            })
        return result

    # Finalban 阶段：从已选英雄中推荐禁用
    if phase == 'finalban':
        if game_state['my_first']:
            candidates = game_state['my_picks'] if len(game_state['my_final_ban']) < 1 else game_state['enemy_picks']
        else:
            candidates = game_state['enemy_picks'] if len(game_state['enemy_final_ban']) < 1 else game_state['my_picks']

        recs = []
        for hero in candidates:
            info = get_hero_info(hero)
            recs.append({
                'hero_code': hero,
                'hero_name': info.get('name', hero),
                'grade': info.get('grade', '0'),
                'job_cd': info.get('job_cd', ''),
                'attribute_cd': info.get('attribute_cd', ''),
                'score': 500,
                'win_rate': 50.0,
                'reason': '从已选英雄中禁用',
                'model': 'finalban'
            })
        return recs[:count]

    # 其他情况：回退到随机推荐
    return get_statistical_recommendations(count)

def get_statistical_recommendations(count=5):
    """统计推荐方法（备用）"""
    phase = game_state['phase']
    recs = []

    my_picks_idx = [hero_to_idx.get(h, -1) for h in game_state['my_picks'] if h in hero_to_idx]
    enemy_picks_idx = [hero_to_idx.get(h, -1) for h in game_state['enemy_picks'] if h in hero_to_idx]

    if phase == 'finalban':
        # Finalban 阶段：推荐禁用对方核心英雄（随机推荐）
        if game_state['my_first']:
            candidates = game_state['my_picks'] if len(game_state['my_final_ban']) < 1 else game_state['enemy_picks']
        else:
            candidates = game_state['enemy_picks'] if len(game_state['enemy_final_ban']) < 1 else game_state['my_picks']

        for hero in candidates:
            info = get_hero_info(hero)
            recs.append({
                'hero_code': hero,
                'hero_name': info.get('name', hero),
                'grade': info.get('grade', '0'),
                'job_cd': info.get('job_cd', ''),
                'attribute_cd': info.get('attribute_cd', ''),
                'score': 500,
                'win_rate': 50.0,
                'reason': '备用推荐',
            })
        recs.sort(key=lambda x: x['score'], reverse=True)
        return recs[:count]

    elif phase == 'preban':
        # Preban 阶段：随机推荐一些英雄
        import random
        shuffled = hero_list.copy()
        random.shuffle(shuffled)
        for hero in shuffled[:count]:
            info = get_hero_info(hero)
            recs.append({
                'hero_code': hero,
                'hero_name': info.get('name', hero),
                'grade': info.get('grade', '0'),
                'job_cd': info.get('job_cd', ''),
                'attribute_cd': info.get('attribute_cd', ''),
                'score': 500,
                'ban_rate': 0,
                'pick_rate': 0,
                'win_rate': 50.0,
                'picks': 0,
            })
        return recs[:count]

    else:
        # Pick 阶段：随机推荐可用英雄
        available = get_available()
        if not available:
            return []
        
        import random
        shuffled = available.copy()
        random.shuffle(shuffled)
        
        for hero in shuffled[:count]:
            info = get_hero_info(hero)
            recs.append({
                'hero_code': hero,
                'hero_name': info.get('name', hero),
                'grade': info.get('grade', '0'),
                'job_cd': info.get('job_cd', ''),
                'attribute_cd': info.get('attribute_cd', ''),
                'score': 500,
                'win_rate': 50.0,
                'probability': 0,
                'picks': 0,
                'model': 'random'
            })
        return recs[:count]


def next_phase():
    """进入下一阶段"""
    phase = game_state['phase']
    if not phase.startswith('pick'):
        return
    
    pn = int(phase.replace('pick', ''))
    mc = len(game_state['my_picks'])
    ec = len(game_state['enemy_picks'])
    
    if game_state['my_first']:
        targets = {1: (1, 0), 2: (1, 2), 3: (3, 2), 4: (3, 4), 5: (5, 5)}
    else:
        targets = {1: (0, 1), 2: (2, 1), 3: (2, 3), 4: (4, 3), 5: (5, 5)}
    
    target = targets.get(pn, (0, 0))
    mt, et = target
    
    if mc >= mt and ec >= et:
        if pn < 5:
            game_state['phase'] = f'pick{pn+1}'
            game_state['step'] += 1
            print(f"[阶段] 进入 {game_state['phase']} (我{mc} 敌{ec})")
        else:
            game_state['phase'] = 'finalban'
            game_state['step'] += 1
            print(f"[阶段] 进入 finalban (我{mc} 敌{ec})")

def get_turn():
    """判断轮到谁"""
    phase = game_state['phase']
    mc = len(game_state['my_picks'])
    ec = len(game_state['enemy_picks'])
    
    if phase == 'preban':
        if game_state['my_first']:
            if len(game_state['my_preban']) < 2:
                return 'my'
            elif len(game_state['enemy_preban']) < 2:
                return 'enemy'
            else:
                return 'next'
        else:
            if len(game_state['enemy_preban']) < 2:
                return 'enemy'
            elif len(game_state['my_preban']) < 2:
                return 'my'
            else:
                return 'next'
    
    if phase == 'finalban':
        if game_state['my_first']:
            if len(game_state['my_final_ban']) < 1:
                return 'my'
            elif len(game_state['enemy_final_ban']) < 1:
                return 'enemy'
            else:
                return 'done'
        else:
            if len(game_state['enemy_final_ban']) < 1:
                return 'enemy'
            elif len(game_state['my_final_ban']) < 1:
                return 'my'
            else:
                return 'done'
    
    if phase.startswith('pick'):
        pn = int(phase.replace('pick', ''))
        
        if game_state['my_first']:
            if pn == 1:
                return 'my' if mc < 1 else ('enemy' if ec < 2 else 'next')
            elif pn == 2:
                return 'enemy' if ec < 2 else 'next'
            elif pn == 3:
                return 'my' if mc < 3 else ('enemy' if ec < 4 else 'next')
            elif pn == 4:
                return 'enemy' if ec < 4 else 'next'
            elif pn == 5:
                return 'my' if mc < 5 else ('enemy' if ec < 5 else 'next')
        else:
            if pn == 1:
                return 'enemy' if ec < 1 else 'next'
            elif pn == 2:
                return 'my' if mc < 2 else ('enemy' if ec < 3 else 'next')
            elif pn == 3:
                return 'enemy' if ec < 3 else 'next'
            elif pn == 4:
                return 'my' if mc < 4 else ('enemy' if ec < 5 else 'next')
            elif pn == 5:
                return 'enemy' if ec < 5 else ('my' if mc < 5 else 'next')
    
    return 'done'

# ==================== 加载 Transformer 模型 ====================
print("\n加载 Transformer 模型...")
recommender = DraftRecommender(model_path=TRANSFORMER_MODEL, hero_list_path='hero_list.json')
use_transformer = recommender.model is not None
if use_transformer:
    print("✅ 使用 Transformer 模型进行推荐")
else:
    print("⚠️ Transformer 模型不可用，回退到统计推荐")

# ==================== HTML 界面 ====================
HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>E7 选秀辅助</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:Arial,sans-serif;background:#f5f5f5;color:#333;font-size:13px}
        .app{display:flex;min-height:100vh}
        .left-panel{width:320px;background:#fff;border-right:1px solid #ddd;padding:15px}
        .center-panel{flex:1;background:#fff;padding:15px;border-right:1px solid #ddd}
        .right-panel{width:260px;background:#fff;padding:15px}
        .panel-title{font-size:14px;color:#666;margin-bottom:12px;font-weight:600}
        .control-box,.info-box,.slots-box,.picks-box,.search-filter-box{background:#f9f9f9;border:1px solid #e0e0e0;border-radius:4px;padding:12px;margin-bottom:15px}
        .btn-group{display:flex;gap:8px;flex-wrap:wrap}
        .btn{padding:8px 14px;border:1px solid #ddd;background:#fff;border-radius:4px;cursor:pointer;font-size:12px;transition:all 0.2s}
        .btn:hover{background:#f0f0f0}
        .btn.active{background:#4CAF50;color:#fff;border-color:#4CAF50}
        .btn-undo{background:#ff9800;color:#fff;border-color:#ff9800}
        .btn-undo:hover{background:#f57c00}
        .turn-indicator{padding:10px;border-radius:4px;text-align:center;margin-bottom:15px;font-weight:600;font-size:14px}
        .turn-indicator.my{background:#4CAF50;color:#fff}
        .turn-indicator.enemy{background:#f44336;color:#fff}
        .turn-indicator.done{background:#9E9E9E;color:#fff}
        .phase-badge{display:inline-block;background:#2196F3;color:#fff;padding:4px 10px;border-radius:12px;font-size:12px;margin-bottom:10px}
        .slots-row{display:flex;gap:10px}.slots-col{flex:1}
        .slots-title{text-align:center;font-size:12px;color:#666;margin-bottom:8px}
        .slots-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
        .slot{width:100%;aspect-ratio:1;border:2px dashed #ddd;border-radius:4px;background:#fafafa;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:11px;color:#999;position:relative}
        .slot.filled{border-style:solid;border-color:#4CAF50;background:#e8f5e9}
        .slot.enemy{border-color:#f44336;background:#ffebee}
        .slot img{width:80%;height:80%;object-fit:contain}
        .slot .hero-name{position:absolute;bottom:-16px;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
        .picks-row{display:flex;gap:15px}.picks-col{flex:1}
        .picks-grid{display:flex;flex-direction:column;gap:6px}
        .pick-slot{width:100%;height:45px;border:2px dashed #ddd;border-radius:4px;background:#fafafa;display:flex;align-items:center;justify-content:center;position:relative}
        .pick-slot.filled{border-style:solid;border-color:#2196F3;background:#e3f2fd}
        .pick-slot.enemy{border-style:solid;border-color:#f44336;background:#ffebee}
        .pick-slot img{width:36px;height:36px;object-fit:contain}
        .pick-slot .hero-name{position:absolute;right:4px;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:60%}
        .search-filter-box{margin-bottom:10px}
        .filter-label{font-size:12px;color:#666;margin-bottom:6px;font-weight:600}
        .filter-buttons{display:flex;gap:6px;flex-wrap:wrap}
        .filter-btn{padding:6px 12px;border:1px solid #ddd;background:#fff;border-radius:4px;cursor:pointer;font-size:11px;transition:all 0.2s}
        .filter-btn:hover{background:#f0f0f0;border-color:#aaa}
        .filter-btn.active{background:#4CAF50;color:#fff;border-color:#4CAF50}
        .filter-btn.fire.active{background:#f44336;color:#fff;border-color:#f44336}
        .filter-btn.ice.active{background:#2196F3;color:#fff;border-color:#2196F3}
        .filter-btn.wind.active{background:#4CAF50;color:#fff;border-color:#4CAF50}
        .filter-btn.dark.active{background:#333;color:#fff;border-color:#333}
        .filter-btn.light.active{background:#fff;color:#333;border-color:#999}
        .search-box{width:100%;padding:8px;border:1px solid #ddd;border-radius:4px;margin-bottom:8px;font-size:12px}
        .hero-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(44px,1fr));gap:6px;max-height:500px;overflow-y:auto;padding:6px}
        .hero-item{width:44px;height:44px;border-radius:50%;overflow:hidden;cursor:pointer;border:2px solid transparent;transition:all 0.2s;position:relative}
        .hero-item:hover{transform:scale(1.1);border-color:#4CAF50}
        .hero-item.disabled{opacity:0.3;cursor:not-allowed}
        .hero-item img{width:100%;height:100%;object-fit:cover}
        .hero-item .grade{position:absolute;top:0;right:0;background:rgba(0,0,0,0.7);color:#fff;font-size:8px;padding:1px 3px;border-radius:3px 0 5px 0}
        .rec-list{display:flex;flex-direction:column;gap:8px}
        .rec-item{display:flex;align-items:center;gap:8px;padding:8px;background:#f9f9f9;border-radius:8px;cursor:pointer;transition:all 0.2s}
        .rec-item:hover{background:#f0f0f0;transform:translateX(5px)}
        .rec-img{width:36px;height:36px;border-radius:50%;overflow:hidden;flex-shrink:0}
        .rec-img img{width:100%;height:100%;object-fit:cover}
        .rec-info{flex:1;min-width:0}
        .rec-name{font-size:11px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;gap:4px}
        .rec-grade{background:#666;color:#fff;padding:1px 3px;border-radius:2px;font-size:9px}
        .rec-grade.grade-5{background:#f44336}
        .rec-grade.grade-4{background:#ff9800}
        .rec-grade.grade-3{background:#4CAF50}
        .rec-job{font-size:9px;color:#999}
        .rec-attr{font-size:9px;padding:1px 3px;border-radius:2px}
        .rec-attr.fire{color:#f44336}
        .rec-attr.ice{color:#2196F3}
        .rec-attr.wind{color:#4CAF50}
        .rec-attr.dark{background:#333;color:#fff}
        .rec-attr.light{background:#fff;color:#333;border:1px solid #ddd}
        .rec-score{font-size:11px;color:#4CAF50;font-weight:600}
        .rec-stats{font-size:9px;color:#999;margin-top:3px;display:flex;gap:6px;flex-wrap:wrap}
        .rec-stat{white-space:nowrap}
        .rec-stat strong{color:#333}
        .history-count{font-size:11px;color:#666;margin-left:8px}
    </style>
</head>
<body>
    <div class="app">
        <div class="left-panel">
            <div class="panel-title">控制</div>
            <div class="control-box">
                <div class="btn-group" style="margin-bottom:8px">
                    <button class="btn" id="myFirstBtn" onclick="setFirst('my')">我方先手</button>
                    <button class="btn" id="enemyFirstBtn" onclick="setFirst('enemy')">对方先手</button>
                </div>
                <div class="btn-group">
                    <button class="btn btn-undo" onclick="undo()" id="undoBtn" disabled>↩ 退回上一步</button>
                </div>
            </div>
            <div class="phase-badge" id="phaseBadge">preban</div>
            <div id="turnIndicator" class="turn-indicator my">轮到：我方</div>
            <div class="info-box">
                <div class="info-row" style="margin:6px 0;display:flex;justify-content:space-between">
                    <span class="info-label">全局步：</span><span class="info-value" id="stepDisplay">0</span>
                </div>
                <div class="info-row" style="margin:6px 0;display:flex;justify-content:space-between">
                    <span class="info-label">先手：</span><span class="info-value" id="firstDisplay">我方</span>
                </div>
                <div class="info-row" style="margin:6px 0;display:flex;justify-content:space-between">
                    <span class="info-label">历史步数：</span><span class="info-value" id="historyCount">0</span>
                </div>
            </div>
            <div class="slots-box">
                <div class="slots-title" style="margin-bottom:10px;">预选禁用</div>
                <div class="slots-row">
                    <div class="slots-col"><div class="slots-title">我方</div><div class="slots-grid" id="myPreban"></div></div>
                    <div class="slots-col"><div class="slots-title">对方</div><div class="slots-grid" id="enemyPreban"></div></div>
                </div>
            </div>
            <div class="picks-box">
                <div class="slots-title" style="margin-bottom:10px;">英雄选择</div>
                <div class="picks-row">
                    <div class="picks-col"><div class="slots-title">我方</div><div class="picks-grid" id="myPicks"></div></div>
                    <div class="picks-col"><div class="slots-title">对方</div><div class="picks-grid" id="enemyPicks"></div></div>
                </div>
            </div>
            <div class="slots-box" id="finalBanBox" style="display:none;">
                <div class="slots-title" style="margin-bottom:10px;">最后禁用</div>
                <div class="slots-row">
                    <div class="slots-col"><div class="slots-title">我方</div><div class="slots-grid" id="myFinalBan"></div></div>
                    <div class="slots-col"><div class="slots-title">对方</div><div class="slots-grid" id="enemyFinalBan"></div></div>
                </div>
            </div>
        </div>
        <div class="center-panel">
            <div class="panel-title">搜索与筛选</div>
            <div class="search-filter-box">
                <input type="text" class="search-box" id="nameFilter" placeholder="搜索英雄名称或代码..." oninput="renderGrid()" style="margin-bottom:10px">
                
                <div style="margin-bottom:8px">
                    <div class="filter-label">星级</div>
                    <div class="filter-buttons" id="gradeFilter">
                        <button class="filter-btn active" data-value="" onclick="setFilter('grade', '')">全部</button>
                        <button class="filter-btn" data-value="5" onclick="setFilter('grade', '5')">⭐⭐⭐⭐⭐ 5 星</button>
                        <button class="filter-btn" data-value="4" onclick="setFilter('grade', '4')">⭐⭐⭐⭐ 4 星</button>
                        <button class="filter-btn" data-value="3" onclick="setFilter('grade', '3')">⭐⭐⭐ 3 星</button>
                    </div>
                </div>
                
                <div style="margin-bottom:8px">
                    <div class="filter-label">职业</div>
                    <div class="filter-buttons" id="jobFilter">
                        <button class="filter-btn active" data-value="" onclick="setFilter('job', '')">全部</button>
                        <button class="filter-btn" data-value="knight" onclick="setFilter('job', 'knight')">骑士</button>
                        <button class="filter-btn" data-value="warrior" onclick="setFilter('job', 'warrior')">战士</button>
                        <button class="filter-btn" data-value="ranger" onclick="setFilter('job', 'ranger')">射手</button>
                        <button class="filter-btn" data-value="assassin" onclick="setFilter('job', 'assassin')">刺客</button>
                        <button class="filter-btn" data-value="mage" onclick="setFilter('job', 'mage')">法师</button>
                        <button class="filter-btn" data-value="manauser" onclick="setFilter('job', 'manauser')">魔导士</button>
                    </div>
                </div>
                
                <div style="margin-bottom:8px">
                    <div class="filter-label">属性</div>
                    <div class="filter-buttons" id="attrFilter">
                        <button class="filter-btn active" data-value="" onclick="setFilter('attr', '')">全部</button>
                        <button class="filter-btn fire" data-value="fire" onclick="setFilter('attr', 'fire')">🔥火</button>
                        <button class="filter-btn ice" data-value="ice" onclick="setFilter('attr', 'ice')">💧水</button>
                        <button class="filter-btn wind" data-value="wind" onclick="setFilter('attr', 'wind')">🌿木</button>
                        <button class="filter-btn dark" data-value="dark" onclick="setFilter('attr', 'dark')">⚫暗</button>
                        <button class="filter-btn light" data-value="light" onclick="setFilter('attr', 'light')">⚪光</button>
                    </div>
                </div>
                
                <button class="btn" onclick="clearFilters()" style="width:100%;margin-top:8px">清空筛选</button>
            </div>
            <div class="panel-title" style="margin-top:15px">英雄池 <span id="heroCount" class="history-count"></span></div>
            <div class="hero-grid" id="heroGrid"></div>
        </div>
        <div class="right-panel">
            <div class="panel-title"><span id="recType">推荐选人</span></div>
            <div class="rec-list" id="recList"></div>
        </div>
    </div>
    <script>
        let state = {myFirst:true,phase:'preban',step:0,myPreban:[],enemyPreban:[],myPicks:[],enemyPicks:[],myFinalBan:[],enemyFinalBan:[],heroes:[]};

        async function init() {
            const res = await fetch('/api/heroes');
            state.heroes = await res.json();
            updateUI(); render(); getRecs();
        }

        async function setFirst(side) {
            if (!confirm('确定重新开始？')) return;
            state.myFirst = side === 'my';
            state.phase = 'preban'; state.step = 0;
            state.myPreban = []; state.enemyPreban = [];
            state.myPicks = []; state.enemyPicks = [];
            state.myFinalBan = []; state.enemyFinalBan = [];
            await fetch('/api/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({myFirst:state.myFirst})});
            updateUI(); render(); getRecs();
        }

        async function undo() {
            try {
                const res = await fetch('/api/undo', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                const result = await res.json();
                if (result.success) {
                    state.myPreban = result.state.my_preban || [];
                    state.enemyPreban = result.state.enemy_preban || [];
                    state.myPicks = result.state.my_picks || [];
                    state.enemyPicks = result.state.enemy_picks || [];
                    state.myFinalBan = result.state.my_final_ban || [];
                    state.enemyFinalBan = result.state.enemy_final_ban || [];
                    state.phase = result.state.phase;
                    state.step = result.state.step;
                    state.historyCount = result.state.history_count || 0;
                    updateUI(); render(); getRecs();
                } else {
                    alert(result.error || '无法退回');
                }
            } catch (e) {
                alert('退回失败：' + e.message);
            }
        }

        // 退回到指定我方 Pick 之前的状态（包括取消那个 Pick）
        async function undoToMyPick(index) {
            const heroCode = state.myPicks[index];
            console.log('=== undoToMyPick 开始 ===');
            console.log('index:', index, 'heroCode:', heroCode);
            console.log('state.myPicks:', state.myPicks);
            console.log('state.historyCount:', state.historyCount);
            
            if (!confirm(`确定要退回到选择 ${heroCode} 之前吗？`)) return;
            
            // 目标步数：4 个 Ban + index 个 Pick（不包含 index 位置的 Pick）
            const targetStep = 4 + index;
            // 计算需要退回的步数
            const steps = state.historyCount - targetStep;
            
            console.log('targetStep:', targetStep, 'steps:', steps);
            
            // 一步步退回
            for (let i = 0; i < steps; i++) {
                console.log('--- 执行第', i+1, '次 undo ---');
                console.log('undo 前 historyCount:', state.historyCount, 'myPicks:', state.myPicks);
                await undo();
                console.log('undo 后 historyCount:', state.historyCount, 'myPicks:', state.myPicks);
            }
            
            console.log('=== undoToMyPick 完成 ===');
            console.log('最终 historyCount:', state.historyCount, 'myPicks:', state.myPicks);
        }

        // 退回到指定敌方 Pick 之前的状态（包括取消那个 Pick）
        async function undoToEnemyPick(index) {
            const heroCode = state.enemyPicks[index];
            console.log('=== undoToEnemyPick 开始 ===');
            console.log('index:', index, 'heroCode:', heroCode);
            console.log('state.enemyPicks:', state.enemyPicks);
            console.log('state.historyCount:', state.historyCount);
            
            if (!confirm(`确定要退回到选择 ${heroCode} 之前吗？`)) return;
            
            const targetStep = 4 + index + 1;
            const steps = state.historyCount - targetStep;
            
            console.log('targetStep:', targetStep, 'steps:', steps);
            
            for (let i = 0; i < steps; i++) {
                console.log('--- 执行第', i+1, '次 undo ---');
                console.log('undo 前 historyCount:', state.historyCount, 'enemyPicks:', state.enemyPicks);
                await undo();
                console.log('undo 后 historyCount:', state.historyCount, 'enemyPicks:', state.enemyPicks);
            }
            
            console.log('=== undoToEnemyPick 完成 ===');
            console.log('最终 historyCount:', state.historyCount, 'enemyPicks:', state.enemyPicks);
        }

        function updateUI() {
            document.getElementById('phaseBadge').textContent = state.phase;
            document.getElementById('stepDisplay').textContent = state.step;
            document.getElementById('firstDisplay').textContent = state.myFirst ? '我方' : '对方';
            document.getElementById('historyCount').textContent = state.historyCount || 0;
            document.getElementById('myFirstBtn').classList.toggle('active', state.myFirst);
            document.getElementById('enemyFirstBtn').classList.toggle('active', !state.myFirst);
            document.getElementById('undoBtn').disabled = (state.historyCount || 0) === 0;
            document.getElementById('finalBanBox').style.display = state.phase === 'finalban' ? 'block' : 'none';
        }

        function getImageUrl(code) {
            // 优先使用本地图片（更快）
            // state.heroes 已经包含了 image_url（本地路径或网络路径）
            const hero = state.heroes.find(x => x.code === code);
            if (hero && hero.image_url) {
                // 如果是本地路径（/hero_images/开头），直接使用
                if (hero.image_url.startsWith('/hero_images/')) {
                    return hero.image_url;
                }
                // 如果是网络路径，添加缓存 busting 参数
                return hero.image_url + '?v=1';
            }
            // 备用：官网 URL
            return 'https://static-pubcomm.onstove.com/event/live/epic7/guide/images/hero/' + code + '_s.png';
        }

        function render() {
            document.getElementById('myPreban').innerHTML = [0,1].map(i => {
                const h = state.myPreban[i];
                const info = state.heroes.find(x => x.code === h) || {code:h,name:h,grade:'0',job_cd:'',attribute_cd:''};
                return '<div class="slot '+(h?'filled':'')+'" onclick="'+(h?'removePreban(\\'my\\','+i+')':'')+'">'+
                    (h?'<img src="'+getImageUrl(h)+'"><span class="hero-name">'+info.name+'</span>':'空')+'</div>';
            }).join('');
            document.getElementById('enemyPreban').innerHTML = [0,1].map(i => {
                const h = state.enemyPreban[i];
                const info = state.heroes.find(x => x.code === h) || {code:h,name:h,grade:'0',job_cd:'',attribute_cd:''};
                return '<div class="slot '+(h?'filled enemy':'')+'" onclick="'+(h?'removePreban(\\'enemy\\','+i+')':'')+'">'+
                    (h?'<img src="'+getImageUrl(h)+'"><span class="hero-name">'+info.name+'</span>':'空')+'</div>';
            }).join('');
            document.getElementById('myPicks').innerHTML = [0,1,2,3,4].map(i => {
                const h = state.myPicks[i];
                const info = state.heroes.find(x => x.code === h) || {code:h,name:h,grade:'0',job_cd:'',attribute_cd:''};
                return '<div class="pick-slot '+(h?'filled':'')+'" onclick="'+(h?'undoToMyPick('+i+')':'')+'">'+
                    (h?'<img src="'+getImageUrl(h)+'"><span class="hero-name">'+info.name+'</span>':'')+'</div>';
            }).join('');
            document.getElementById('enemyPicks').innerHTML = [0,1,2,3,4].map(i => {
                const h = state.enemyPicks[i];
                const info = state.heroes.find(x => x.code === h) || {code:h,name:h,grade:'0',job_cd:'',attribute_cd:''};
                return '<div class="pick-slot '+(h?'filled enemy':'')+'" onclick="'+(h?'undoToEnemyPick('+i+')':'')+'">'+
                    (h?'<img src="'+getImageUrl(h)+'"><span class="hero-name">'+info.name+'</span>':'')+'</div>';
            }).join('');
            document.getElementById('myFinalBan').innerHTML = [0].map(i => {
                const h = state.myFinalBan[i];
                const info = state.heroes.find(x => x.code === h) || {code:h,name:h,grade:'0',job_cd:'',attribute_cd:''};
                return '<div class="slot '+(h?'filled':'')+'">'+
                    (h?'<img src="'+getImageUrl(h)+'"><span class="hero-name">'+info.name+'</span>':'空')+'</div>';
            }).join('');
            document.getElementById('enemyFinalBan').innerHTML = [0].map(i => {
                const h = state.enemyFinalBan[i];
                const info = state.heroes.find(x => x.code === h) || {code:h,name:h,grade:'0',job_cd:'',attribute_cd:''};
                return '<div class="slot '+(h?'filled enemy':'')+'" onclick="'+(h?'removeFinalBan(\\'enemy\\','+i+')':'')+'">'+
                    (h?'<img src="'+getImageUrl(h)+'"><span class="hero-name">'+info.name+'</span>':'空')+'</div>';
            }).join('');
            renderGrid(); updateTurn();
        }

        function updateTurn() {
            fetch('/api/turn').then(r => r.json()).then(turn => {
                const ind = document.getElementById('turnIndicator');
                if (turn === 'my') { ind.textContent = '轮到：我方'; ind.className = 'turn-indicator my'; }
                else if (turn === 'enemy') { ind.textContent = '轮到：对方'; ind.className = 'turn-indicator enemy'; }
                else if (turn === 'done') { ind.textContent = '选秀结束'; ind.className = 'turn-indicator done'; }
                else { ind.textContent = '进入下一阶段'; ind.className = 'turn-indicator my'; }
            });
        }

        // 筛选状态
        let filters = {grade: '', job: '', attr: ''};

        function setFilter(type, value) {
            filters[type] = value;
            // 更新按钮状态
            const container = document.getElementById(type + 'Filter');
            container.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.value === value);
            });
            renderGrid();
        }

        function clearFilters() {
            filters = {grade: '', job: '', attr: ''};
            document.getElementById('nameFilter').value = '';
            ['grade', 'job', 'attr'].forEach(type => {
                const container = document.getElementById(type + 'Filter');
                container.querySelectorAll('.filter-btn').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.value === '');
                });
            });
            renderGrid();
        }

        function renderGrid() {
            const banned = new Set([...state.myPreban, ...state.enemyPreban]);
            const used = new Set([...state.myPicks, ...state.enemyPicks]);

            const nameQ = (document.getElementById('nameFilter').value || '').toLowerCase();
            const gradeQ = filters.grade;
            const jobQ = filters.job;
            const attrQ = filters.attr;

            let heroes = state.heroes;

            console.log('renderGrid: phase=', state.phase, 'myPicks=', state.myPicks, 'enemyPicks=', state.enemyPicks, 'banned=', [...banned]);

            if (state.phase === 'finalban') {
                heroes = state.myFirst ? (state.myFinalBan.length < 1 ? state.myPicks : state.enemyPicks) : (state.enemyFinalBan.length < 1 ? state.enemyPicks : state.myPicks);
                heroes = heroes.map(code => state.heroes.find(h => h.code === code) || {code, name:code, grade:'0', job_cd:'', attribute_cd:''});
            } else if (state.phase.startsWith('pick')) {
                // Pick 阶段：过滤已选和已 Ban 的英雄
                heroes = state.heroes.filter(h => !used.has(h.code) && !banned.has(h.code));
            }
            // Preban 阶段：不过滤英雄（可以重复 Ban）
            // 添加调试信息
            console.log('renderGrid: heroes count=', heroes.length, 'used=', [...used]);

            // 应用筛选
            if (nameQ) heroes = heroes.filter(h => h.name.toLowerCase().includes(nameQ) || h.code.toLowerCase().includes(nameQ));
            if (gradeQ) heroes = heroes.filter(h => h.grade === gradeQ);
            if (jobQ) heroes = heroes.filter(h => h.job_cd === jobQ);
            if (attrQ) heroes = heroes.filter(h => h.attribute_cd === attrQ);

            document.getElementById('heroCount').textContent = heroes.length + ' / ' + state.heroes.length;

            fetch('/api/turn').then(r => r.json()).then(turn => {
                const can = turn !== 'done' && turn !== 'next';
                document.getElementById('heroGrid').innerHTML = heroes.map(h => {
                    const dis = state.phase.startsWith('pick') && (used.has(h.code) || banned.has(h.code));
                    return '<div class="hero-item '+(dis?'disabled':'')+'" onclick="'+(dis||!can?'':'select(\\''+h.code+'\\')')+'">'+
                        '<img src="'+getImageUrl(h.code)+'">'+
                        '<span class="grade">'+h.grade+'</span></div>';
                }).join('');
            });
        }

        async function select(code) {
            const banned = new Set([...state.myPreban, ...state.enemyPreban]);
            const used = new Set([...state.myPicks, ...state.enemyPicks]);
            // Pick 阶段：不能选已选或已 Ban 的英雄
            // Preban 阶段：可以重复 Ban（不检查）
            if (state.phase.startsWith('pick') && (used.has(code) || banned.has(code))) return;

            const turn = await fetch('/api/turn').then(r => r.json());
            if (turn === 'done' || turn === 'next') return;

            const action = state.phase === 'preban' ? 'preban' : (state.phase === 'finalban' ? 'finalban' : 'pick');
            try {
                const res = await fetch('/api/action', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,hero:code,side:turn})});
                const result = await res.json();
                if (result.success) {
                    state.myPreban = result.state.my_preban || [];
                    state.enemyPreban = result.state.enemy_preban || [];
                    state.myPicks = result.state.my_picks || [];
                    state.enemyPicks = result.state.enemy_picks || [];
                    state.myFinalBan = result.state.my_final_ban || [];
                    state.enemyFinalBan = result.state.enemy_final_ban || [];
                    state.phase = result.state.phase;
                    state.step = result.state.step;
                    state.historyCount = result.state.history_count || 0;
                    updateUI(); render(); getRecs();
                } else {
                    alert(result.error || '操作失败');
                }
            } catch (e) {
                alert('选择失败：' + e.message);
            }
        }

        function removePreban(side, i) {
            if (side === 'my') state.myPreban.splice(i, 1);
            else state.enemyPreban.splice(i, 1);
            render();
        }

        function removeFinalBan(side, i) {
            if (side === 'my') state.myFinalBan.splice(i, 1);
            else state.enemyFinalBan.splice(i, 1);
            render();
        }

        async function getRecs() {
            const res = await fetch('/api/recommend');
            const recs = await res.json();

            const phase = state.phase;
            const banned = new Set([...state.myPreban, ...state.enemyPreban]);
            
            let typeText = '';
            if (phase === 'preban') typeText = '推荐禁用';
            else if (phase === 'pick1') typeText = '推荐一选';
            else if (phase === 'finalban') typeText = '推荐禁用 (从已选 5 人中)';
            else typeText = '推荐选取';

            document.getElementById('recType').textContent = typeText + ' (' + recs.length + ')';

            if (Array.isArray(recs) && recs.length > 0) {
                document.getElementById('recList').innerHTML = recs.map((r, i) => {
                    const gradeClass = r.grade === '5' ? 'grade-5' : (r.grade === '4' ? 'grade-4' : 'grade-3');
                    const attrClass = r.attribute_cd || '';
                    // Preban 阶段：推荐列表永远可点击（因为对方可能也要 Ban 同一个）
                    // Pick 阶段：已选/已 Ban 的英雄不可点
                    const isUsed = phase !== 'preban' && banned.has(r.hero_code);
                    return '<div class="rec-item" style="'+(isUsed?'opacity:0.3;cursor:not-allowed;':'')+'" onclick="'+(isUsed?'':'select(\\''+r.hero_code+'\\')')+'">'+
                        '<div class="rec-img"><img src="'+getImageUrl(r.hero_code)+'"></div>'+
                        '<div class="rec-info">'+
                            '<div class="rec-name">'+
                                '<span class="rec-grade '+gradeClass+'">'+r.grade+'</span>'+
                                r.hero_name+
                                (r.job_cd ? '<span class="rec-job">'+r.job_cd+'</span>' : '')+
                                (r.attribute_cd ? '<span class="rec-attr '+attrClass+'">'+r.attribute_cd+'</span>' : '')+
                            '</div>'+
                            '<div class="rec-score">'+r.score+'分</div>'+
                            '<div class="rec-stats">'+
                                (r.win_rate ? '<span class="rec-stat"><strong>胜率</strong>'+r.win_rate+'%</span>' : '')+
                                (r.probability ? '<span class="rec-stat"><strong>概率</strong>'+r.probability+'%</span>' : '')+
                                (r.duo_score !== undefined ? '<span class="rec-stat"><strong>配合</strong>'+r.duo_score+'%</span>' : '')+
                                (r.counter_score !== undefined ? '<span class="rec-stat"><strong>克制</strong>'+r.counter_score+'%</span>' : '')+
                            '</div>'+
                        '</div>'+
                    '</div>';
                }).join('');
            } else {
                document.getElementById('recList').innerHTML = '<p style="color:#999;text-align:center;">暂无推荐</p>';
            }
        }

        init();
    </script>
</body>
</html>
'''

# ==================== HTTP 服务器 ====================
class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}]: {format % args}")

    def do_GET(self):
        try:
            # 提取路径（去除查询字符串）
            path = self.path.split('?')[0]

            # 处理本地英雄图片请求
            if path.startswith('/hero_images/'):
                filename = path.replace('/hero_images/', '')
                image_path = os.path.join(HERO_IMAGES_DIR, filename)
                
                if os.path.exists(image_path):
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Cache-Control', 'public, max-age=86400')  # 缓存 1 天
                    self.end_headers()
                    
                    with open(image_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'Image not found')
                return

            if path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(HTML.encode('utf-8'))
            elif path == '/api/heroes':
                # 返回包含完整信息的英雄列表
                heroes = []
                for code in hero_list:
                    info = get_hero_info(code)
                    info['image_url'] = get_hero_image_url(code)  # 添加本地图片 URL
                    heroes.append(info)
                self.send_json(heroes)
            elif path == '/api/recommend':
                self.send_json(get_recommendations())
            elif path == '/api/turn':
                self.send_json(get_turn())
            else:
                self.send_response(404)
        except Exception as e:
            print(f"GET 错误：{e}")
            self.send_response(500)

    def do_POST(self):
        try:
            # 提取路径（去除查询字符串）
            path = self.path.split('?')[0]
            
            # 读取请求体（如果有）
            length = self.headers.get('Content-Length')
            if length:
                data = json.loads(self.rfile.read(int(length)))
            else:
                data = {}

            if path == '/api/reset':
                reset_game(data.get('myFirst', True))
                self.send_json({'success': True})
                return

            if path == '/api/undo':
                success = undo_last_action()
                if success:
                    self.send_json({
                        'success': True,
                        'state': {
                            'my_preban': game_state['my_preban'],
                            'enemy_preban': game_state['enemy_preban'],
                            'my_picks': game_state['my_picks'],
                            'enemy_picks': game_state['enemy_picks'],
                            'my_final_ban': game_state['my_final_ban'],
                            'enemy_final_ban': game_state['enemy_final_ban'],
                            'phase': game_state['phase'],
                            'step': game_state['step'],
                            'history_count': len(game_state['history'])
                        }
                    })
                else:
                    self.send_json({'success': False, 'error': '没有可退回的步骤'})
                return

            action = data.get('action')
            hero = data.get('hero')
            side = data.get('side', 'my')

            print(f"[动作] {action} - {hero} - {side} - 阶段:{game_state['phase']} 我{len(game_state['my_picks'])} 敌{len(game_state['enemy_picks'])}")

            if action == 'preban':
                # 保存当前状态到历史
                save_state()
                
                if side == 'my':
                    if hero in game_state['my_preban']:
                        self.send_json({'success': False, 'error': '不能重复禁用'})
                        return
                    if len(game_state['my_preban']) < 2:
                        game_state['my_preban'].append(hero)
                        if len(game_state['my_preban']) >= 2 and len(game_state['enemy_preban']) >= 2:
                            game_state['phase'] = 'pick1'
                            game_state['step'] = 1
                            print("  → 进入 pick1")
                elif side == 'enemy':
                    if hero in game_state['enemy_preban']:
                        self.send_json({'success': False, 'error': '不能重复禁用'})
                        return
                    if len(game_state['enemy_preban']) < 2:
                        game_state['enemy_preban'].append(hero)
                        if len(game_state['enemy_preban']) >= 2 and len(game_state['my_preban']) >= 2:
                            game_state['phase'] = 'pick1'
                            game_state['step'] = 1
                            print("  → 进入 pick1")

            elif action == 'pick':
                # 保存当前状态到历史
                save_state()
                
                used = set(game_state['my_picks'] + game_state['enemy_picks'])
                banned = set(game_state['my_preban'] + game_state['enemy_preban'])
                if hero in used or hero in banned:
                    self.send_json({'success': False, 'error': '英雄不可用'})
                    return
                if side == 'my' and len(game_state['my_picks']) < 5:
                    game_state['my_picks'].append(hero)
                elif side == 'enemy' and len(game_state['enemy_picks']) < 5:
                    game_state['enemy_picks'].append(hero)
                next_phase()

            elif action == 'finalban':
                # 保存当前状态到历史
                save_state()
                
                if side == 'my' and len(game_state['my_final_ban']) < 1:
                    game_state['my_final_ban'].append(hero)
                elif side == 'enemy' and len(game_state['enemy_final_ban']) < 1:
                    game_state['enemy_final_ban'].append(hero)
                    game_state['phase'] = 'done'

            self.send_json({'success': True, 'state': {
                'my_preban': game_state['my_preban'],
                'enemy_preban': game_state['enemy_preban'],
                'my_picks': game_state['my_picks'],
                'enemy_picks': game_state['enemy_picks'],
                'my_final_ban': game_state['my_final_ban'],
                'enemy_final_ban': game_state['enemy_final_ban'],
                'phase': game_state['phase'],
                'step': game_state['step'],
                'history_count': len(game_state['history'])
            }})
        except Exception as e:
            print(f"POST 错误：{e}")
            import traceback
            traceback.print_exc()
            self.send_response(500)

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

if __name__ == '__main__':
    print(f"\n🌐 http://localhost:{PORT}")
    print("\n📖 流程 (双方各选 5 个):")
    print("   我方先手：preban(我 2→敌 2) → pick1(我 1) → pick2(敌 2) → pick3(我 2) → pick4(敌 2) → pick5(敌 1) → finalban")
    print("   对方先手：preban(敌 2→我 2) → pick1(敌 1) → pick2(我 2) → pick3(敌 2) → pick4(我 2) → pick5(我 1) → finalban")
    print("\n✨ 新功能:")
    print("   - 按名称/星级/职业/属性筛选英雄")
    print("   - 退回上一步重新选择")
    print("   - e7.json 完整映射")
    print("\n⚠️ Ctrl+C 停止")
    print("="*70)
    try:
        print(f"\n✅ 服务器已启动")
        HTTPServer(('localhost', PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 停止")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
