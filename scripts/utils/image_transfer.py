import os
import json
from datetime import time

import pandas as pd
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def oss_download_images(
        pkl_file_path: str = 'D:/receive/dxm/data/flux_coco_2.pkl',
        user_json_path: str = "D:/receive/dxm/data/user.json",
        user_start_idx: int = 90,
        user_end_idx: int = 1000,
        local_save_base_dir: str = "F:/flux_coco_1000_user/",
        oss_bucket: str = "gai-algorithm",  # 需要替换为实际bucket name
        oss_remote_base_path: str = "user/wangzihao/129/disk4/nfs-55/data/flux_coco_49/",  # OSS中的基础路径，对应原remote_path
        negative: bool = False,
):
    """
    使用ossutil从OSS远程下载图像到本地

    参数:
    - pkl_file_path: pkl文件路径
    - user_json_path: user.json文件路径
    - user_start_idx: 用户ID起始索引
    - user_end_idx: 用户ID结束索引
    - local_save_base_dir: 本地保存基础目录
    - oss_endpoint: OSS endpoint
    - oss_bucket: OSS bucket名称
    - oss_remote_base_path: OSS中远程基础路径

    注意: 需要提前配置好ossutil，确保命令行可以执行ossutil命令
    """

    # 1. 加载 pkl 文件和user.json
    logger.info("正在加载数据文件...")
    try:
        data = pd.read_pickle(pkl_file_path)
        with open(user_json_path, "rb") as f:
            user_list = json.load(f)
        logger.info(f"成功加载数据: {len(data)}条记录, {len(user_list)}个用户")
    except Exception as e:
        logger.error(f"加载数据文件失败: {str(e)}")
        raise

    # 2. 获取用户ID列表
    user_index_list = user_list[user_start_idx:user_end_idx]
    logger.info(f"将处理 {len(user_index_list)} 个用户 (索引 {user_start_idx} 到 {user_end_idx})")

    # 3. 确保本地保存目录存在
    local_save_dir = Path(local_save_base_dir)
    local_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"本地保存目录: {local_save_dir}")

    # 4. 遍历用户
    total_downloaded = 0
    failed_downloads = []

    for i, user_id in enumerate(user_index_list, 1):
        logger.info(f"处理用户 {i}/{len(user_index_list)}: {user_id}")

        try:
            # 获取该用户的数据
            if user_id not in data:
                logger.warning(f"用户ID {user_id} 在数据中不存在，跳过")
                continue

            image_data = data[user_id][:-1]  # 排除最后一个元素

            # 5. 遍历该用户的所有图像对
            for j, image_item in enumerate(image_data, 1):
                logger.info(f"  处理图像对 {j}/{len(image_data)}")

                # 构建OSS远程路径和本地保存路径

                positive_image_remote = f"oss://{oss_bucket}/{oss_remote_base_path}{image_item[1]}"

                positive_image_local = str(local_save_dir / Path(image_item[1]).name)
                if negative:
                    negative_image_remote = f"oss://{oss_bucket}/{oss_remote_base_path}{image_item[0]}"
                    negative_image_local = str(local_save_dir / Path(image_item[0]).name)
                    # 6. 下载负样本图像
                    logger.info(f"    下载负样本: {image_item[0]}")
                    if download_from_oss(
                            remote_path=negative_image_remote,
                            local_path=negative_image_local,
                    ):
                        total_downloaded += 1
                    else:
                        failed_downloads.append(f"negative_{user_id}_{image_item[0]}")

                # 7. 下载正样本图像
                logger.info(f"    下载正样本: {image_item[1]}")
                if download_from_oss(
                        remote_path=positive_image_remote,
                        local_path=positive_image_local,
                ):
                    total_downloaded += 1
                else:
                    failed_downloads.append(f"positive_{user_id}_{image_item[1]}")

        except Exception as e:
            logger.error(f"处理用户 {user_id} 时出错: {str(e)}")
            continue

    # 8. 总结结果
    logger.info(f"下载完成! 总共下载: {total_downloaded} 个文件")
    if failed_downloads:
        logger.warning(f"下载失败的文件: {len(failed_downloads)} 个")
        logger.warning(f"失败列表: {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")

    return {
        "total_downloaded": total_downloaded,
        "failed_downloads": failed_downloads,
        "success_rate": total_downloaded / (total_downloaded + len(failed_downloads)) if (total_downloaded + len(
            failed_downloads)) > 0 else 0
    }


def oss_transfer_images(
        pkl_file_path: str = 'D:/receive/dxm/data/flux_coco_2.pkl',
        user_json_path: str = "D:/receive/dxm/data/user.json",
        user_start_idx: int = 0,
        user_end_idx: int = 1000,
        remote_save_base_dir: str = "oss://gai-algorithm/user/wangzihao/129/disk4/nfs-55/data/flux_coco_1000_user",
        oss_bucket: str = "gai-algorithm",  # 需要替换为实际bucket name
        oss_remote_base_path: str = "user/wangzihao/129/disk4/nfs-55/data/flux_coco_49/",  # OSS中的基础路径，对应原remote_path
        negative: bool = False
):
    """
    使用ossutil从OSS远程下载图像到本地

    参数:
    - pkl_file_path: pkl文件路径
    - user_json_path: user.json文件路径
    - user_start_idx: 用户ID起始索引
    - user_end_idx: 用户ID结束索引
    - local_save_base_dir: 本地保存基础目录
    - oss_endpoint: OSS endpoint
    - oss_bucket: OSS bucket名称
    - oss_remote_base_path: OSS中远程基础路径

    注意: 需要提前配置好ossutil，确保命令行可以执行ossutil命令
    """

    # 1. 加载 pkl 文件和user.json
    logger.info("正在加载数据文件...")
    try:
        data = pd.read_pickle(pkl_file_path)
        with open(user_json_path, "rb") as f:
            user_list = json.load(f)
        logger.info(f"成功加载数据: {len(data)}条记录, {len(user_list)}个用户")
    except Exception as e:
        logger.error(f"加载数据文件失败: {str(e)}")
        raise

    # 2. 获取用户ID列表
    # user_index_list = user_list[user_start_idx:user_end_idx]
    user_index_list = set()
    with open("D:/Lab/Premier/log/untransferred_files.json") as f:
        user_data = json.load(f)
    for data_dict in user_data:
        user_index_list.add(data_dict["user_id"])
    logger.info(f"将处理 {len(user_index_list)} 个用户 (索引 {user_start_idx} 到 {user_end_idx})")

    # 3. 确保本地保存目录存在

    logger.info(f"远程保存目录: {remote_save_base_dir}")

    # 4. 遍历用户
    total_downloaded = 0
    failed_downloads = []

    for i, user_id in enumerate(user_index_list, 1):
        logger.info(f"处理用户 {i}/{len(user_index_list)}: {user_id}")

        try:
            # 获取该用户的数据
            if user_id not in data:
                logger.warning(f"用户ID {user_id} 在数据中不存在，跳过")
                continue

            image_data = data[user_id][:-1]  # 排除最后一个元素

            # 5. 遍历该用户的所有图像对
            for j, image_item in enumerate(image_data, 1):
                logger.info(f"  处理图像对 {j}/{len(image_data)}")

                # 构建OSS远程路径和本地保存路径

                positive_image_remote = f"oss://{oss_bucket}/{oss_remote_base_path}{image_item[1]}"

                positive_image_local = f"{remote_save_base_dir}/"
                if negative:
                    negative_image_remote = f"oss://{oss_bucket}/{oss_remote_base_path}{image_item[0]}"
                    negative_image_local = f"{remote_save_base_dir}/"
                    # 6. 下载负样本图像
                    logger.info(f"    下载负样本: {image_item[0]}")
                    if download_from_oss(
                            remote_path=negative_image_remote,
                            local_path=negative_image_local,
                    ):
                        total_downloaded += 1
                    else:
                        failed_downloads.append(f"negative_{user_id}_{image_item[0]}")

                # 7. 下载正样本图像
                logger.info(f"    下载正样本: {image_item[1]}")
                if download_from_oss(
                        remote_path=positive_image_remote,
                        local_path=positive_image_local,
                ):
                    total_downloaded += 1
                else:
                    failed_downloads.append(f"positive_{user_id}_{image_item[1]}")

        except Exception as e:
            logger.error(f"处理用户 {user_id} 时出错: {str(e)}")
            continue

    # 8. 总结结果
    logger.info(f"下载完成! 总共下载: {total_downloaded} 个文件")
    if failed_downloads:
        logger.warning(f"下载失败的文件: {len(failed_downloads)} 个")
        logger.warning(f"失败列表: {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")

    return {
        "total_downloaded": total_downloaded,
        "failed_downloads": failed_downloads,
        "success_rate": total_downloaded / (total_downloaded + len(failed_downloads)) if (total_downloaded + len(
            failed_downloads)) > 0 else 0
    }


def download_from_oss(
        remote_path: str,
        local_path: str,
        max_retries: int = 3
) -> bool:
    """
    使用ossutil下载单个文件

    参数:
    - remote_path: OSS远程路径 (格式: oss://bucket/path/to/file)
    - local_path: 本地保存路径
    - endpoint: OSS endpoint
    - max_retries: 最大重试次数

    返回:
    - bool: 是否下载成功
    """

    # 确保本地目录存在
    # Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # 构建ossutil命令
    command = [
        "ossutil", "cp", "-f",
        remote_path,
        local_path,
    ]

    logger.info(f"执行命令: {' '.join(command)}")

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
                logger.info(f"✅ 下载成功: {local_path}")
                return True
            else:
                logger.warning(f"⚠️  下载失败 (尝试 {attempt + 1}/{max_retries}): {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  命令执行失败 (尝试 {attempt + 1}/{max_retries}): {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️  下载超时 (尝试 {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.warning(f"⚠️  未知错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

    logger.error(f"❌ 下载失败，已达到最大重试次数: {Path(local_path).name}")
    return False


# 使用示例
if __name__ == "__main__":
    # 配置OSS参数 (需要替换为实际值)
    OSS_CONFIG = {
        "oss_endpoint": "oss-cn-beijing.aliyuncs.com",  # 替换为你的OSS endpoint
        "oss_bucket": "your-bucket-name",  # 替换为你的bucket name
        "oss_remote_base_path": "data/flux_coco_2/"  # OSS中的基础路径
    }

    try:
        result = oss_transfer_images()

        print("\n" + "=" * 50)
        print(f"最终结果:")
        print(f"成功下载: {result['total_downloaded']} 个文件")
        print(f"失败下载: {len(result['failed_downloads'])} 个文件")
        print(f"成功率: {result['success_rate']:.2%}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"主程序执行失败: {str(e)}")
