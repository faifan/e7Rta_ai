"""
一键训练流水线
  准备：先确认 e7.json 已放好
  步骤：1_get_data → 2_clean_data_146 → 3_train_transformer_v2 → 部署到 D:\e7Rta_zd
"""
import os
import sys
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DST  = r"D:\e7Rta_zd"

DEPLOY_FILES = [
    "draft_transformer.pth",
    "hero_list_146.json",
]

STEPS = [
    ("采集对战数据",   "1_get_data.py"),
    ("清洗数据",       "2_clean_data_146.py"),
    ("训练模型",       "3_train_transformer_v2.py"),
]


def check_ready():
    e7 = os.path.join(HERE, 'e7.json')
    if not os.path.exists(e7):
        print("✗ 未找到 e7.json，请先准备好英雄列表再运行。")
        sys.exit(1)
    print("✓ e7.json 已就绪")


def run_step(label, script):
    path = os.path.join(HERE, script)
    print(f"\n{'='*60}")
    print(f"  {label}  ({script})")
    print(f"{'='*60}")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, path], cwd=HERE)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print(f"\n⏹ 收到停止信号，正在终止 {script}...")
        proc.terminate()
        proc.wait()
        print("已停止。")
        sys.exit(0)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n✗ {label} 失败（exit {proc.returncode}），流水线中断。")
        sys.exit(proc.returncode)
    print(f"\n✓ {label} 完成  耗时 {elapsed:.1f}s")


def deploy():
    print(f"\n{'='*60}")
    print(f"  部署到 {DST}")
    print(f"{'='*60}")
    for fname in DEPLOY_FILES:
        src = os.path.join(HERE, fname)
        dst = os.path.join(DST, fname)
        if not os.path.exists(src):
            print(f"  [SKIP] {fname}  (文件不存在)")
            continue
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        print(f"  [OK]   {fname}  ({size/1024/1024:.2f} MB)")


if __name__ == '__main__':
    print("=" * 60)
    print("  E7 RTA 训练流水线")
    print("=" * 60)
    check_ready()

    total_start = time.time()
    for label, script in STEPS:
        run_step(label, script)

    deploy()

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  全部完成！总耗时 {total/60:.1f} 分钟")
    print(f"{'='*60}")
