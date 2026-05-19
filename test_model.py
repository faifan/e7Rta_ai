"""
第七史诗选秀辅助 - 模型测试脚本
用于验证 .pth 模型文件是否能正常加载和推理
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import json
from model import DraftTransformer


def test_model_loading(model_path='draft_transformer.pth'):
    """测试 1: 模型加载"""
    print("=" * 60)
    print("测试 1: 模型加载")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✅ 模型文件加载成功")
        print(f"   - 文件大小：{os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
        print(f"   - 包含内容：{list(checkpoint.keys())}")
        
        # 显示模型配置
        if 'config' in checkpoint:
            print(f"\n模型配置:")
            for k, v in checkpoint['config'].items():
                print(f"   - {k}: {v}")
        
        # 显示训练信息
        if 'val_acc' in checkpoint:
            print(f"\n训练结果:")
            print(f"   - 验证准确率：{checkpoint.get('val_acc', 0):.4f}")
            print(f"   - 训练轮数：{checkpoint.get('epoch', 0)}")
        
        return checkpoint
    
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return None


def test_model_inference(checkpoint, hero_list_path='hero_list.json'):
    """测试 2: 模型推理"""
    print("\n" + "=" * 60)
    print("测试 2: 模型推理")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载英雄列表
    if not os.path.exists(hero_list_path):
        print(f"❌ 英雄列表文件不存在：{hero_list_path}")
        return
    
    with open(hero_list_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        hero_list = data['hero_list']
        hero_to_idx = data['hero_to_idx']
        idx_to_hero = {i: h for i, h in enumerate(hero_list)}
    
    print(f"英雄数量：{len(hero_list)}")
    
    # 创建模型
    config = checkpoint['config']
    model = DraftTransformer(
        num_heroes=config['num_heroes'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型创建成功")
    print(f"   - 参数量：{sum(p.numel() for p in model.parameters()) / 1000000:.2f}M")
    
    # 测试推理
    print(f"\n进行推理测试...")
    
    # 模拟输入：空序列（Preban 阶段）
    batch_size = 1
    seq_len = 5
    
    hero_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    side_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    phase_ids = torch.tensor([0], dtype=torch.long, device=device)  # Preban 阶段
    src_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    # 打印输入数据
    print(f"\n输入数据:")
    print(f"  hero_ids 形状：{hero_ids.shape}, 值：{hero_ids.tolist()}")
    print(f"  side_ids 形状：{side_ids.shape}, 值：{side_ids.tolist()}")
    print(f"  phase_ids 形状：{phase_ids.shape}, 值：{phase_ids.tolist()}")
    print(f"  src_mask 形状：{src_mask.shape}, 值：{src_mask.tolist()}")
    
    try:
        with torch.no_grad():
            logits, win_rate = model(hero_ids, side_ids, phase_ids, src_mask)
        
        print(f"✅ 推理成功")
        print(f"   - 输出 logits 形状：{logits.shape}")
        print(f"   - 胜率预测形状：{win_rate.shape}")
        print(f"   - 预测胜率：{win_rate.item():.4f}")
        
        # 获取 Top-5 推荐
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, 5)
        
        print(f"\nTop-5 推荐英雄:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            hero_code = idx_to_hero.get(idx.item(), 'unknown')
            print(f"   {i}. {hero_code} - 概率：{prob.item():.4f}")
        
        return model
    
    except Exception as e:
        print(f"❌ 推理失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline(model_path='draft_transformer.pth'):
    """测试 3: 完整流程（使用 DraftRecommender）"""
    print("\n" + "=" * 60)
    print("测试 3: 完整推荐流程")
    print("=" * 60)
    
    try:
        from transformer_inference import DraftRecommender
        
        recommender = DraftRecommender(model_path=model_path)
        
        if recommender.model is None:
            print("❌ 推荐器初始化失败")
            return
        
        # 测试 Preban 推荐
        print("\n测试 Preban 推荐:")
        recs = recommender.recommend(
            my_picks=[],
            enemy_picks=[],
            banned=[],
            phase='preban',
            my_first=True,
            top_k=5
        )
        
        for i, rec in enumerate(recs, 1):
            print(f"   {i}. {rec['hero_code']} - 概率：{rec['probability']:.4f}")
        
        # 测试 Pick1 推荐
        print("\n测试 Pick1 推荐 (假设我方先手):")
        recs = recommender.recommend(
            my_picks=[],
            enemy_picks=[],
            banned=['c1153', 'c1133'],  # 假设 Ban 了光呆和暗呆
            phase='pick1',
            my_first=True,
            top_k=5
        )
        
        for i, rec in enumerate(recs, 1):
            print(f"   {i}. {rec['hero_code']} - 概率：{rec['probability']:.4f}, 胜率：{rec['win_rate']:.4f}")
        
        # 测试 Pick2 推荐
        print("\n测试 Pick2 推荐 (我方已选 1 个，敌方已选 2 个):")
        recs = recommender.recommend(
            my_picks=['c1002'],  # 假设选了赛西莉亚
            enemy_picks=['c1067', 'c1053'],  # 假设敌方选了塔玛林尔、巴萨尔
            banned=['c1153', 'c1133', 'c1117', 'c1155'],
            phase='pick2',
            my_first=True,
            top_k=5
        )
        
        for i, rec in enumerate(recs, 1):
            print(f"   {i}. {rec['hero_code']} - 概率：{rec['probability']:.4f}, 胜率：{rec['win_rate']:.4f}")
        
        print("\n✅ 完整流程测试通过!")
        
    except Exception as e:
        print(f"❌ 完整流程测试失败：{e}")
        import traceback
        traceback.print_exc()


def test_cuda_availability():
    """测试 CUDA 可用性"""
    print("=" * 60)
    print("CUDA 环境检查")
    print("=" * 60)
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 可用：{torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本：{torch.version.cuda}")
        print(f"GPU 数量：{torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("将使用 CPU 进行推理")
    
    print()


# ==================== 主函数 ====================
if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "第七史诗选秀辅助 - 模型测试" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 1. CUDA 检查
    test_cuda_availability()
    
    # 2. 模型加载测试
    checkpoint = test_model_loading('draft_transformer.pth')
    
    if checkpoint is None:
        print("\n❌ 模型加载失败，无法继续测试")
        exit(1)
    
    # 3. 模型推理测试
    model = test_model_inference(checkpoint)
    
    if model is None:
        print("\n❌ 模型推理失败，无法继续测试")
        exit(1)
    
    # 4. 完整流程测试
    test_full_pipeline('draft_transformer.pth')
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成!")
    print("=" * 60)
