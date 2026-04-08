#!/bin/bash

#export http_proxy=http://127.0.0.1:7890
#export https_proxy=http://127.0.0.1:7890
# shellcheck disable=SC2088
#export HF_HOME="~/.cache/huggingface/hub"

# *[Specify the config file path]
export OMINI_CONFIG="train/config/premier_user.yaml"
# *[Specify the WANDB API key]
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'
echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=360000
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_IB_RETRY_CNT=13
# 参数设置
MIN_FREE_MEMORY=40000   # 默认显存阈值40000
GPU_NUM=1
# 函数：get_gpus_by_memory
# 参数：
#   $1 - 所需的最小显存（单位MB）
#   $2 - 需要的GPU数量
# 返回值：
#   标准输出：找到的GPU设备ID（空格分隔）
#   退出码：0-成功，1-未找到足够的显卡或命令失败
get_gpus_by_memory() {
    local min_memory=$1
    local required_gpus=$2
    local gpu_ids=()

    # 验证输入
    if ! [[ "$min_memory" =~ ^[0-9]+$ ]] || ! [[ "$required_gpus" =~ ^[0-9]+$ ]]; then
        echo "错误：参数必须是正整数" >&2
        return 1
    fi

    # 检查nvidia-smi是否存在
    if ! command -v nvidia-smi &> /dev/null; then
        echo "错误：未找到 nvidia-smi 命令" >&2
        return 1
    fi

    # 获取GPU信息
    while IFS=, read -r index memory_free; do
        # 清理数字并转换为整数
        index=$(echo "$index" | tr -d ' ')
        memory_free=$(echo "$memory_free" | awk '{print $1}')

        # 检查显存是否满足条件
        if [[ "$memory_free" =~ ^[0-9]+$ ]] && (( memory_free >= min_memory )) && [[ $index != 2 ]]; then
            gpu_ids+=("$index")
            # 找到足够数量的GPU即停止
            if [[ ${#gpu_ids[@]} -eq $required_gpus ]]; then
                break
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)

    # 检查结果
    if [[ ${#gpu_ids[@]} -lt $required_gpus ]]; then
#        echo "错误：仅找到 ${#gpu_ids[@]} 块显存 >= ${min_memory}MB 的显卡（需要${required_gpus}块）" >&2
        return 1
    fi

    echo "${gpu_ids[@]}"
    return 0
}
execute_func() {
  accelerate launch --main_process_port 10806 -m scripts.train_flux.train_user_embedding
#  python omini/train_flux/train_personalize_delta.py
  return $?
}

# 主程序
while true; do
#    echo "[$(date +%H:%M:%S)] 检测GPU显存..."
    # 获取符合条件的GPU
    if ! gpus_string=$(get_gpus_by_memory "${MIN_FREE_MEMORY}" "${GPU_NUM}"); then
        continue
    fi
    read -ra gpu_ids <<< "$gpus_string"
    # 将GPU ID转为逗号分隔的字符串（CUDA_VISIBLE_DEVICES格式）
    cuda_devices=$(IFS=,; echo "${gpu_ids[*]}")

    echo "使用 GPU(s): ${cuda_devices}"
    echo "执行命令: $*"

    # 设置可见GPU并执行命令
    export CUDA_VISIBLE_DEVICES="$cuda_devices"

#    execute_func
    execute_func
    # 执行完成后退出监控
    exit_code=$?
    echo "exit code ${exit_code}"

    if [[ $exit_code -ne 110 ]]; then
      break
    fi
done
