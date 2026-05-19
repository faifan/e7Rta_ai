"""
下载英雄图片脚本

从第七史诗官网下载所有英雄图片到本地 hero_images 目录
并生成映射文件 hero_images_mapping.json
"""

import os
import json
import urllib.request
from pathlib import Path

# 配置
HERO_LIST_FILE = 'hero_list.json'
HERO_IMAGES_DIR = 'hero_images'
HERO_IMAGES_MAPPING = 'hero_images_mapping.json'

# 官网图片 URL 模板
IMAGE_URL_TEMPLATE = 'https://static-pubcomm.onstove.com/event/live/epic7/guide/images/hero/{}_s.png'


def download_image(code, output_path):
    """下载单张英雄图片"""
    url = IMAGE_URL_TEMPLATE.format(code)
    
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  ✗ 下载失败 {code}: {e}")
        return False


def main():
    print("="*70)
    print("第七史诗 - 下载英雄图片")
    print("="*70)
    
    # 加载英雄列表
    print(f"\n加载英雄列表：{HERO_LIST_FILE}")
    if not os.path.exists(HERO_LIST_FILE):
        print(f"❌ 找不到 {HERO_LIST_FILE}")
        print("请先运行 1_get_data.py 生成英雄列表")
        return
    
    with open(HERO_LIST_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        hero_list = data['hero_list']
    
    print(f"英雄总数：{len(hero_list)}")
    
    # 创建图片目录
    os.makedirs(HERO_IMAGES_DIR, exist_ok=True)
    
    # 下载图片
    print(f"\n开始下载图片到 {HERO_IMAGES_DIR}/")
    print("-"*70)
    
    mapping = {}
    success = 0
    failed = 0
    
    for i, code in enumerate(hero_list, 1):
        # 文件名：code.png
        filename = f"{code}.png"
        output_path = os.path.join(HERO_IMAGES_DIR, filename)
        
        # 跳过已存在的图片
        if os.path.exists(output_path):
            mapping[code] = filename
            success += 1
            print(f"[{i}/{len(hero_list)}] ✓ 已存在 {code}")
            continue
        
        # 下载
        print(f"[{i}/{len(hero_list)}] 下载 {code}...", end=' ', flush=True)
        if download_image(code, output_path):
            mapping[code] = filename
            success += 1
            print("✓")
        else:
            failed += 1
    
    # 保存映射
    print("\n" + "-"*70)
    print(f"保存映射：{HERO_IMAGES_MAPPING}")
    
    with open(HERO_IMAGES_MAPPING, 'w', encoding='utf-8') as f:
        json.dump({
            'mapping': mapping,
            'total': len(hero_list),
            'success': success,
            'failed': failed
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 下载完成！")
    print(f"  成功：{success}/{len(hero_list)}")
    print(f"  失败：{failed}/{len(hero_list)}")
    print(f"  图片目录：{HERO_IMAGES_DIR}/")
    print(f"  映射文件：{HERO_IMAGES_MAPPING}")
    
    print("\n" + "="*70)
    print("使用说明")
    print("="*70)
    print("""
3_start_web.py 会自动检测本地图片：
- 如果找到 hero_images/ 和 hero_images_mapping.json
- 将使用本地图片（加载更快）
- 否则使用官网图片
""")


if __name__ == '__main__':
    main()
