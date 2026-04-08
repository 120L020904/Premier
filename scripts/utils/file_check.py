import os
import json
import pandas as pd
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time


def check_untransferred_files(
        pkl_file_path: str = 'D:/receive/dxm/data/flux_coco_2.pkl',
        user_json_path: str = "D:/receive/dxm/data/user.json",
        user_start_idx: int = 0,
        user_end_idx: int = 1000,
        oss_bucket: str = "gai-algorithm",
        oss_remote_base_path: str = "user/wangzihao/129/disk4/nfs-55/data/flux_coco_1000_user/",
) -> List[Dict]:
    """
    检查哪些文件还没有被传输到OSS，返回未传输文件列表

    参数:
    - pkl_file_path: pkl文件路径
    - user_json_path: user.json文件路径
    - user_start_idx: 用户ID起始索引
    - user_end_idx: 用户ID结束索引
    - local_base_path: 本地文件基础路径
    - oss_bucket: OSS bucket名称
    - oss_remote_base_path: OSS远程基础路径
    - oss_endpoint: OSS endpoint

    返回:
    - 未传输文件列表，每个元素包含: user_id, file_type, local_path, remote_path
    """

    print("=== 开始检查未传输文件 ===")
    print(f"加载数据文件: {pkl_file_path}")
    print(f"用户范围: {user_start_idx} 到 {user_end_idx}")

    # 1. 加载 pkl 文件和user.json
    data = pd.read_pickle(pkl_file_path)
    with open(user_json_path, "rb") as f:
        user_list = json.load(f)

    print(f"成功加载数据: {len(data)}条记录, {len(user_list)}个用户")

    # 2. 获取用户ID列表
    user_index_list = user_list[user_start_idx:user_end_idx]
    print(f"将检查 {len(user_index_list)} 个用户的文件")

    # 3. 构建所有需要检查的文件列表
    all_files_to_check = []
    total_files = 0

    print("\n构建文件列表...")
    for user_id in user_index_list:
        if user_id not in data:
            print(f"警告: 用户ID {user_id} 在数据中不存在，跳过")
            continue

        image_data = data[user_id][:-1]  # 排除最后一个元素

        for image_item in image_data:
            positive_image_name = image_item[1]

            # 构建OSS远程路径
            negative_remote_path = f"{oss_remote_base_path}{image_item[0]}"
            positive_remote_path = f"{oss_remote_base_path}{image_item[1]}"
            if image_item[1]=="0000041848_0260063700.png":
                print(f"user_id is {user_id}")
            # 添加到检查列表
            # all_files_to_check.append({
            #     'user_id': user_id,
            #     'file_type': 'negative',
            #     'local_path': negative_image_name,
            #     'remote_path': negative_remote_path,
            #     'filename': image_item[0]
            # })

            all_files_to_check.append({
                'user_id': user_id,
                'file_type': 'positive',
                'local_path': positive_image_name,
                'remote_path': positive_remote_path,
                'filename': image_item[1]
            })

            total_files += 1

    print(f"总共需要检查的文件数: {total_files}")

    # 4. 获取OSS上已存在的文件列表
    print("\n获取OSS上已存在的文件列表...")
    existing_files = get_oss_existing_files(oss_bucket, oss_remote_base_path,)

    if not existing_files:
        print("警告: 无法获取OSS文件列表，假设所有文件都未传输")
        return all_files_to_check

    print(f"OSS上已存在文件数: {len(existing_files)}")

    # 5. 检查哪些文件未传输
    print("\n检查未传输文件...")
    untransferred_files = []

    for file_info in all_files_to_check:
        remote_filename = file_info['local_path']

        # 检查文件是否存在于OSS
        if remote_filename not in existing_files:
            untransferred_files.append(file_info)

    print(f"未传输文件数: {len(untransferred_files)}")

    # 6. 生成报告
    print("\n=== 未传输文件统计 ===")
    user_stats = {}
    for file_info in untransferred_files:
        user_id = file_info['user_id']
        user_stats[user_id] = user_stats.get(user_id, 0) + 1

    print(f"涉及用户数: {len(user_stats)}")
    print("前5个用户未传输文件数:")
    for i, (user_id, count) in enumerate(sorted(user_stats.items())[:5], 1):
        print(f"  {i}. 用户 {user_id}: {count} 个文件")

    if len(user_stats) > 5:
        print(f"  ... 其他 {len(user_stats) - 5} 个用户")

    # 7. 保存未传输文件列表到文件
    output_file = "D:/Lab/Premier/log/untransferred_files.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(untransferred_files, f, indent=2, ensure_ascii=False)

    print(f"\n未传输文件列表已保存至: {output_file}")
    print(f"文件格式示例: {json.dumps(untransferred_files[0] if untransferred_files else {}, indent=2)}")

    return untransferred_files


def get_oss_existing_files(
        bucket_name: str,
        remote_path: str,
        max_retries: int = 3
) -> Dict[str, str]:
    """
    获取OSS指定路径下已存在的文件列表

    参数:
    - bucket_name: OSS bucket名称
    - remote_path: 远程路径
    - endpoint: OSS endpoint
    - max_retries: 最大重试次数

    返回:
    - 文件名到完整路径的映射字典
    """

    # 构建ossutil ls命令
    # 使用 -d 参数只列出目录，但我们需要列出文件，所以不使用-d
    # 使用 --marker 和 --max-keys 来处理大量文件
    command = [
        "ossutil",
        "ls",
        f"oss://{bucket_name}/{remote_path}",

    ]

    print(f"执行OSS命令: {' '.join(command)}")

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                print("✅ 成功获取OSS文件列表")
                return parse_oss_ls_output(result.stdout)
            else:
                print(f"⚠️  OSS命令执行失败 (尝试 {attempt + 1}/{max_retries})")
                print(f"错误输出: {result.stderr[:500]}")

        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            print(f"错误输出: {e.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print(f"❌ 命令超时 (尝试 {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"❌ 未知错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

    print("❌ 达到最大重试次数，无法获取OSS文件列表")
    return {}


def parse_oss_ls_output(output: str) -> Dict[str, str]:
    """
    解析ossutil ls命令的输出，提取文件名

    返回:
    - {文件名: 完整远程路径} 的字典
    """
    existing_files = {}

    # 按行分割输出
    lines = output.strip().split('\n')

    # 跳过标题行（通常包含LastModified等）
    # ossutil ls的输出格式: LastModified   Size   ObjectName
    for line in lines:
        line = line.strip()
        if not line or line.startswith('LastModified') or line.startswith('----------'):
            continue

        # 分割行，通常用空格分隔
        parts = line.split()
        if len(parts) < 3:
            continue

        # 最后一个部分是ObjectName（文件路径）
        object_name = parts[-1]

        # 提取文件名
        filename = object_name.split("/")[-1]
        if filename:  # 确保不是空文件名
            existing_files[filename] = object_name

    print(f"解析到 {len(existing_files)} 个有效文件")
    return existing_files


# 使用示例
if __name__ == "__main__":


    try:
        untransferred_files = check_untransferred_files(
            pkl_file_path='D:/receive/dxm/data/flux_coco_2.pkl',
            user_json_path="D:/receive/dxm/data/user.json",
            user_start_idx=0,
            user_end_idx=1000,

        )

        print(f"\n{'=' * 50}")
        print(f"检查完成! 未传输文件总数: {len(untransferred_files)}")
        print(f"{'=' * 50}")

        # 显示前10个未传输文件
        print("\n前10个未传输文件:")
        for i, file_info in enumerate(untransferred_files[:10], 1):
            print(f"{i}. 用户: {file_info['user_id']}, 类型: {file_info['file_type']}")
            print(f"   本地路径: {file_info['local_path']}")
            print(f"   远程路径: {file_info['remote_path']}")
            print(f"   文件名: {file_info['filename']}")
            print("-" * 50)

        if len(untransferred_files) > 10:
            print(f"... 还有 {len(untransferred_files) - 10} 个文件")
        with open("D:/Premier/log/untransferred_files.json", "w", encoding="utf-8") as f:
            json.dump(untransferred_files, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        print("💡 建议检查:")
        print("- OSS bucket名称、endpoint是否正确")
        print("- ossutil是否已正确配置和安装")
        print("- 网络连接是否正常")
        print("- 文件路径是否存在")