"""
1. 获取数据脚本
从官网批量获取所有服务器前 100 名玩家的战斗记录
"""

import requests
import json
import sys
import os
import time
import concurrent.futures
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置 ====================
SERVERS = [
    {'name': '韩国', 'code': 'world_kor'},
    {'name': '亚洲', 'code': 'world_asia'},
    {'name': '全球', 'code': 'world_global'},
    {'name': '欧洲', 'code': 'world_eu'},
    {'name': '日本', 'code': 'world_jpn'},
]

PLAYERS_PER_SERVER = 100
BATTLE_PAGES = 10
MAX_WORKERS = 20
MAX_RETRIES = 5
RETRY_DELAY = 2

OUTPUT_DIR = 'output'
RANKING_API = 'https://e7api.onstove.com/gameApi/getWorldUserRankingDetail'
BATTLE_API = 'https://e7api.onstove.com/gameApi/getBattleList'
SEASON_CODE = 'pvp_rta_ss19'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

# ==================== 函数 ====================
def get_ranking(world_code, retry=MAX_RETRIES):
    """获取排名"""
    all_players = []
    for page in range(1, 11):
        success = False
        for attempt in range(1, retry + 1):
            try:
                params = {
                    'season_code': SEASON_CODE,
                    'world_code': world_code,
                    'current_page': str(page),
                    'lang': 'zh-TW'
                }
                r = requests.post(RANKING_API, data=params, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('code') == 0:
                        result = data['value'].get('result_body', [])
                        all_players.extend(result)
                        print(f"  第{page}页：{len(result)} 人")
                        success = True
                        if len(result) < 10:
                            break
                        break
                print(f"  第{page}页 失败 ({attempt}/{retry})")
                time.sleep(RETRY_DELAY * attempt)
            except Exception as e:
                print(f"  第{page}页 错误 ({attempt}/{retry}): {e}")
                time.sleep(RETRY_DELAY * attempt)
        if not success:
            print(f"  第{page}页 放弃")
        time.sleep(0.5)
    return all_players[:PLAYERS_PER_SERVER]

def get_battles_with_retry(nick_no, world_code, max_pages=BATTLE_PAGES):
    """获取战斗记录"""
    all_battles = []
    for page in range(1, max_pages + 1):
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                params = {
                    'nick_no': str(nick_no),
                    'world_code': world_code,
                    'current_page': str(page),
                    'lang': 'zh-TW'
                }
                r = requests.post(BATTLE_API, data=params, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('code') == 0:
                        result = data['value'].get('result_body', {})
                        battle_list = result.get('battle_list', [])
                        if battle_list:
                            all_battles.extend(battle_list)
                            success = True
                        if len(battle_list) < 10:
                            break
                        break
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
            except:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
        time.sleep(0.1)
    return nick_no, all_battles

def get_battles_task(args):
    """并发任务"""
    nick_no, world_code, name = args
    print(f"  获取 {name} ({nick_no})...")
    nick_no, battles = get_battles_with_retry(nick_no, world_code)
    print(f"    获取 {len(battles)} 场")
    return nick_no, battles

# ==================== 主程序 ====================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("第七史诗 - 1.获取数据脚本")
    print("="*70)
    print(f"\n服务器：{len(SERVERS)}")
    print(f"每服务器：{PLAYERS_PER_SERVER} 名玩家")
    print(f"每玩家：{BATTLE_PAGES * 10} 场战斗")
    print(f"并发线程：{MAX_WORKERS}")
    print(f"最大重试：{MAX_RETRIES} 次")
    
    start_time = time.time()
    all_data = {'export_time': datetime.now().isoformat(), 'season_code': SEASON_CODE, 'servers': []}
    total_battles = 0
    
    for server in SERVERS:
        sname, scode = server['name'], server['code']
        print(f"\n{'='*70}")
        print(f"获取 {sname} ({scode}) ...")
        print(f"{'='*70}")
        
        server_data = {'name': sname, 'code': scode, 'players': []}
        
        # 获取排名
        print("\n获取排名...")
        players = get_ranking(scode)
        print(f"共获取 {len(players)} 名玩家")
        
        if not players:
            print("⚠️ 排名获取失败，跳过此服务器")
            continue
        
        # 并发获取战斗记录
        print(f"\n并发获取战斗记录 ({MAX_WORKERS} 线程，重试{MAX_RETRIES}次)...")
        
        tasks = [(p.get('nick_no'), scode, p.get('nickname', '?')) for p in players if p.get('nick_no')]
        battle_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(get_battles_task, task): task for task in tasks}
            completed = 0
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    nick_no, battles = future.result()
                    battle_results[nick_no] = battles
                    completed += 1
                    
                    if completed % 10 == 0:
                        total = sum(len(b) for b in battle_results.values())
                        print(f"  进度 {completed}/{len(tasks)} - 累计 {total:,} 场")
                except Exception as e:
                    print(f"  错误：{e}")
        
        # 组装数据
        print("\n组装数据...")
        for i, player in enumerate(players[:PLAYERS_PER_SERVER]):
            nick_no = player.get('nick_no')
            if not nick_no:
                continue
            
            battles = battle_results.get(nick_no, [])
            
            player_data = {
                'rank': player.get('season_rank', i+1),
                'nick_no': nick_no,
                'nickname': player.get('nickname', '?'),
                'server': scode,
                'score': player.get('win_score', 0),
                'win_rate': player.get('win_rate', 0),
                'wins': player.get('win_cnt', 0),
                'losses': player.get('lose_cnt', 0),
                'clan': player.get('clan_name', ''),
                'battle_count': len(battles),
                'battles': battles
            }
            
            server_data['players'].append(player_data)
            total_battles += len(battles)
        
        # 保存
        with open(f'{OUTPUT_DIR}/server_{scode}_final.json', 'w', encoding='utf-8') as f:
            json.dump(server_data, f, ensure_ascii=False, indent=2)
        
        all_data['servers'].append(server_data)
        print(f"\n已保存：server_{scode}_final.json ({len(server_data['players'])} 玩家，{total_battles:,} 场)")
    
    # 保存总文件
    with open(f'{OUTPUT_DIR}/all_complete_fast.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print(f"\n{'='*70}")
    print(f"完成！耗时：{hours}小时 {minutes}分钟")
    print(f"总战斗：{total_battles:,} 场")
    print(f"数据：{OUTPUT_DIR}/all_complete_fast.json")
    print(f"{'='*70}")
