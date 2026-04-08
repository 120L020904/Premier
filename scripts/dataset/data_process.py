import json
import os
from collections import defaultdict
import random
from pathlib import Path


def split_dataset_by_user_and_prompt(
        data_dir: str,
        output_dir: str,
        num_users: int = 50,
        train_ratio: float = 2 / 3,
        random_seed: int = 42
):
    """
    为每个用户单独分割数据集，按prompt字段分组，同一个prompt的数据2/3做训练，1/3做测试

    参数:
    - data_dir: 包含用户数据文件的目录
    - output_dir: 输出目录
    - num_users: 要处理的用户数量（前N个用户）
    - train_ratio: 训练集比例（默认2/3）
    - random_seed: 随机种子，确保可重复性
    """
    # 设置随机种子
    random.seed(random_seed)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取前num_users个用户文件
    user_files = sorted(os.listdir(data_dir))
    user_files = [f for f in user_files if f.endswith('.jsonl') or f.endswith('.json')]
    user_files = user_files[:num_users]

    print(f"找到 {len(user_files)} 个用户文件，将处理前 {num_users} 个:")
    for i, f in enumerate(user_files[:5], 1):
        print(f"  {i}. {f}")
    if len(user_files) > 5:
        print(f"  ... 其他 {len(user_files) - 5} 个文件")

    # 存储所有用户的统计信息
    all_stats = {
        "total_users": len(user_files),
        "user_stats": {},
        "overall_stats": {
            "total_samples": 0,
            "total_train_samples": 0,
            "total_test_samples": 0,
            "total_unique_prompts": 0
        }
    }

    # 处理每个用户文件
    for user_idx, user_file in enumerate(user_files, 1):
        user_id = Path(user_file).stem
        file_path = os.path.join(data_dir, user_file)

        print(f"\n{'=' * 50}")
        print(f"处理用户 {user_idx}/{len(user_files)}: {user_id}")
        print(f"文件路径: {file_path}")

        # 按prompt分组存储该用户的数据
        prompt_groups = defaultdict(list)

        # 读取用户数据
        total_samples = 0
        unique_prompts = set()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if 'prompt' not in data:
                        print(f"  警告: 用户 {user_id} 文件第 {line_num} 行缺少 'prompt' 字段，已跳过")
                        continue

                    prompt = data['prompt'].strip()
                    if not prompt:  # 跳过空prompt
                        print(f"  警告: 用户 {user_id} 文件第 {line_num} 行 'prompt' 字段为空，已跳过")
                        continue

                    prompt_groups[prompt].append(data)
                    unique_prompts.add(prompt)
                    total_samples += 1
                except json.JSONDecodeError as e:
                    print(f"  警告: 用户 {user_id} 文件第 {line_num} 行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"  警告: 用户 {user_id} 文件第 {line_num} 行处理错误: {e}")
                    continue

        print(f"  用户 {user_id} 数据加载完成:")
        print(f"    总样本数: {total_samples}")
        print(f"    唯一prompt数量: {len(unique_prompts)}")

        if total_samples == 0:
            print(f"  警告: 用户 {user_id} 没有有效数据，跳过分割")
            continue

        # 分割该用户的数据集
        train_data = []
        test_data = []

        for prompt, group_data in prompt_groups.items():
            if len(group_data) < 2:
                # 只有一个样本的组，全部放入训练集
                train_data.extend(group_data)
                continue

            # 随机打乱同一prompt下的数据
            random.shuffle(group_data)

            # 按比例分割
            split_idx = int(len(group_data) * train_ratio)
            if split_idx == 0:  # 确保至少有一个样本在训练集
                split_idx = 1
            if split_idx == len(group_data):  # 确保至少有一个样本在测试集
                split_idx -= 1

            train_data.extend(group_data[:split_idx])
            test_data.extend(group_data[split_idx:])

        print(f"  用户 {user_id} 分割结果:")
        print(f"    训练集样本数: {len(train_data)}")
        print(f"    测试集样本数: {len(test_data)}")
        print(f"    总样本数: {len(train_data) + len(test_data)}")

        # 保存该用户的训练集
        train_filename = f"{user_id}_train.jsonl"
        train_file = output_path / train_filename
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 保存该用户的测试集
        test_filename = f"{user_id}_test.jsonl"
        test_file = output_path / test_filename
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"  用户 {user_id} 分割完成:")
        print(f"    训练集保存至: {train_file}")
        print(f"    测试集保存至: {test_file}")

        # 记录统计信息
        user_stats = {
            "user_id": user_id,
            "total_samples": total_samples,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "unique_prompts": len(unique_prompts),
            "train_ratio": len(train_data) / total_samples if total_samples > 0 else 0,
            "test_ratio": len(test_data) / total_samples if total_samples > 0 else 0,
            "train_file": str(train_file),
            "test_file": str(test_file)
        }

        all_stats["user_stats"][user_id] = user_stats
        all_stats["overall_stats"]["total_samples"] += total_samples
        all_stats["overall_stats"]["total_train_samples"] += len(train_data)
        all_stats["overall_stats"]["total_test_samples"] += len(test_data)
        all_stats["overall_stats"]["total_unique_prompts"] += len(unique_prompts)

    # 生成总体统计报告
    overall_stats = all_stats["overall_stats"]
    if overall_stats["total_samples"] > 0:
        overall_stats["overall_train_ratio"] = overall_stats["total_train_samples"] / overall_stats["total_samples"]
        overall_stats["overall_test_ratio"] = overall_stats["total_test_samples"] / overall_stats["total_samples"]

    # 保存统计报告
    report_file = output_path / "split_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print("所有用户分割完成!")
    print(f"总用户数: {all_stats['total_users']}")
    print(f"总样本数: {overall_stats['total_samples']}")
    print(f"总训练样本数: {overall_stats['total_train_samples']} ({overall_stats.get('overall_train_ratio', 0):.1%})")
    print(f"总测试样本数: {overall_stats['total_test_samples']} ({overall_stats.get('overall_test_ratio', 0):.1%})")
    print(f"总唯一prompt数: {overall_stats['total_unique_prompts']}")
    print(f"统计报告已保存至: {report_file}")
    print(f"{'=' * 50}")

    return all_stats


# 使用示例
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "/hdd5/wangzihao/data/PIP-dataset"  # 替换为你的数据集目录
    OUTPUT_DIR = "/hdd5/wangzihao/data/PIP_user_split_dataset"  # 输出目录

    # 执行分割
    stats = split_dataset_by_user_and_prompt(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        num_users=50,
        train_ratio=2 / 3,
        random_seed=42
    )

    # 验证部分用户的结果
    print(f"\n{'=' * 50}")
    print("验证部分用户分割结果:")

    # 获取前3个用户的验证信息
    for i, (user_id, user_stats) in enumerate(list(stats["user_stats"].items())[:3], 1):
        print(f"\n用户 {i}: {user_id}")
        print(f"  总样本数: {user_stats['total_samples']}")
        print(f"  训练集样本数: {user_stats['train_samples']}")
        print(f"  测试集样本数: {user_stats['test_samples']}")

        # 读取训练集前2行
        train_file = user_stats['train_file']
        if os.path.exists(train_file):
            print(f"  训练集文件: {train_file}")
            with open(train_file, 'r', encoding='utf-8') as f:
                for j, line in enumerate(f, 1):
                    if j > 2:
                        break
                    try:
                        data = json.loads(line.strip())
                        print(f"    训练样本 {j}: {data.get('prompt', '')[:50]}...")
                    except:
                        print(f"    训练样本 {j}: 读取失败")

        # 读取测试集前2行
        test_file = user_stats['test_file']
        if os.path.exists(test_file):
            print(f"  测试集文件: {test_file}")
            with open(test_file, 'r', encoding='utf-8') as f:
                for j, line in enumerate(f, 1):
                    if j > 2:
                        break
                    try:
                        data = json.loads(line.strip())
                        print(f"    测试样本 {j}: {data.get('prompt', '')[:50]}...")
                    except:
                        print(f"    测试样本 {j}: 读取失败")

    print(f"\n{'=' * 50}")
    print("分割完成! 每个用户都有独立的训练集和测试集文件")
    print(f"文件命名规则: 原文件名_train.jsonl 和 原文件名_test.jsonl")
    print(f"{'=' * 50}")