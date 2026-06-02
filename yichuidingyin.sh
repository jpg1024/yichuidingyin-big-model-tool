#!/bin/bash

# ==============================================================================
# 一锤定音大模型工具 (CHUIZI)
# 版本：1.1 (增加微调与vLLM选项控制)
# 作者：锤子代码（公众号）
# 功能：交互式下载大模型，生成推理脚本和微调脚本
# ==============================================================================

set -euo pipefail

# 全局变量
MODEL_ID=""
SAVE_PATH=""
DOWNLOAD_OPTION=""
SELECTED_SITE=""
DOWNLOAD_URL=""
MODEL_NAME=""
ORG_NAME=""
MODEL_TYPE=""
DATASETS=""
FINETUNE_OUTPUT_DIR=""
BATCH_SIZE=1
GRAD_ACCUM_STEPS=16
LORA_RANK=8
LORA_ALPHA=32
NUM_EPOCHS=1
MAX_LENGTH=2048
TARGET_MODULES="all-linear"

# 新增控制变量
NEED_FINETUNE="n"
USE_VLLM="n"

# 检查是否安装了必要的工具
check_dependencies() {
    echo -e "正在检查依赖..."
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        echo -e "错误: 未找到 python3。请先安装 Python 3。"
        exit 1
    fi
    
    # 检查 pip
    if ! python3 -m pip --version &> /dev/null; then
        echo -e "错误: 未找到 pip。请先安装 pip。"
        exit 1
    fi
    
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        echo -e "错误: 未找到 wget 或 curl 命令。请先安装其中一个。"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        echo -e "错误: 未找到 git 命令。请先安装 git。"
        exit 1
    fi
    
    # 检查并安装 ms-swift
    if ! python3 -c "import swift" &> /dev/null; then
        echo -e "ms-swift 未安装，正在安装..."
        if ! python3 -m pip install git+https://github.com/modelscope/ms-swift.git --upgrade; then
            echo -e "普通安装失败，尝试使用 --user 选项安装..."
            python3 -m pip install git+https://github.com/modelscope/ms-swift.git --upgrade --user
        fi
    else
        echo -e "ms-swift 已安装"
    fi
    
    # 检查并安装 vllm (仅在需要时安装，但为保持兼容性先保留检查)
    if ! python3 -c "import vllm" &> /dev/null; then
        echo -e "vllm 未安装，正在安装..."
        if ! python3 -m pip install vllm --upgrade; then
            echo -e "普通安装失败，尝试使用 --user 选项安装..."
            python3 -m pip install vllm --upgrade --user
        fi
    else
        echo -e "vllm 已安装"
    fi
    
    echo -e "依赖检查通过\n"
}

# 显示欢迎信息
show_welcome() {
    cat << "EOF"
    
 ___   _  _   _   _   ___   ____  ___   ___      _     ___   __  __     _   
  / __| | || | | | | | |_ _| |_  / |_ _| |   \    /_\   |_ _| |  \/  |   /_\  
 | (__  | __ | | |_| |  | |   / /   | |  | |) |  / _ \   | |  | |\/| |  / _ \ 
  \___| |_||_|  \___/  |___| /___| |___| |___/  /_/ \_\ |___| |_|  |_| /_/ \_\
                                                                              
  
       一锤定音大模型工具 (CHUIZI) - 版本 1.1
       作者：锤子代码（公众号）
       代码仓库：https://github.com/pruidong/yichuidingyin-big-model-tool
  
EOF
    echo -e "欢迎使用一锤定音大模型工具！\n"
}

# 选择模型下载站点
select_download_site() {
    while true; do
        echo -e "请选择模型下载站点："
        echo "1) HF-Mirror 镜像站 (https://hf-mirror.com)"
        echo "2) ModelScope 魔搭社区 (https://www.modelscope.cn)"
        
        read -p "$(echo -e '输入选项 [1-2]: ')" choice
        
        case $choice in
            1)
                SELECTED_SITE="hf-mirror"
                DOWNLOAD_URL="https://hf-mirror.com"
                break
                ;;
            2)
                SELECTED_SITE="modelscope"
                DOWNLOAD_URL="https://www.modelscope.cn"
                break
                ;;
            *)
                echo -e "无效选项，请输入 1 或 2\n"
                ;;
        esac
    done
    
    echo -e "已选择站点: $DOWNLOAD_URL\n"
}

# 获取模型ID
get_model_id() {
    while true; do
        read -p "$(echo -e '请输入模型ID (例如: Qwen/Qwen2.5-7B-Instruct): ')" MODEL_ID
        
        if [[ -z "$MODEL_ID" ]]; then
            echo -e "模型ID不能为空，请重新输入"
            continue
        fi
        
        # 提取组织名和模型名
        if [[ "$MODEL_ID" == */* ]]; then
            ORG_NAME="${MODEL_ID%%/*}"
            MODEL_NAME="${MODEL_ID##*/}"
        else
            ORG_NAME="unknown"
            MODEL_NAME="$MODEL_ID"
        fi
        
        echo -e "已输入模型ID: $MODEL_ID\n"
        break
    done
}

# 获取模型类型
get_model_type() {
    echo -e "请参考以下文档选择正确的模型类型：\n"
    echo -e "https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html"
    echo -e "\n在文档中查找'Model Type'列，选择与您下载的模型匹配的类型"
    
    while true; do
        read -p "$(echo -e '请输入模型类型: ')" MODEL_TYPE
        
        if [[ -z "$MODEL_TYPE" ]]; then
            echo -e "模型类型不能为空，请重新输入"
            continue
        fi
        
        echo -e "已输入模型类型: $MODEL_TYPE\n"
        break
    done
}

# 选择保存路径
select_save_path() {
    echo -e "请选择模型保存路径："
    echo "1) /root/models (默认)"
    echo "2) /root/big-models"
    echo "3) 自定义路径"
    
    while true; do
        read -p "$(echo -e '输入选项 [1-3]: ')" choice
        
        case $choice in
            1)
                SAVE_PATH="/root/models"
                break
                ;;
            2)
                SAVE_PATH="/root/big-models"
                break
                ;;
            3)
                read -p "$(echo -e '请输入自定义路径: ')" custom_path
                if [[ -z "$custom_path" ]]; then
                    echo -e "路径不能为空"
                    continue
                fi
                SAVE_PATH="$custom_path"
                break
                ;;
            *)
                echo -e "无效选项，请输入 1、2 或 3"
                ;;
        esac
    done
    
    # 创建目标目录
    TARGET_DIR="$SAVE_PATH/$MODEL_ID"
    mkdir -p "$TARGET_DIR"
    
    echo -e "模型将保存到: $TARGET_DIR\n"
}

# 选择下载选项
select_download_option() {
    echo -e "请选择下载选项："
    echo "1) 下载全部文件"
    echo "2) 下载指定文件（以逗号分隔）"
    
    while true; do
        read -p "$(echo -e '输入选项 [1-2]: ')" choice
        
        case $choice in
            1)
                DOWNLOAD_OPTION="all"
                break
                ;;
            2)
                DOWNLOAD_OPTION="files"
                read -p "$(echo -e '请输入要下载的文件名（以逗号分隔）: ')" file_list
                if [[ -z "$file_list" ]]; then
                    echo -e "文件列表不能为空"
                    continue
                fi
                break
                ;;
            *)
                echo -e "无效选项，请输入 1 或 2"
                ;;
        esac
    done
    echo
}

# 询问是否微调
ask_finetune_option() {
    echo -e "请问是否需要配置微调参数并生成微调脚本？"
    while true; do
        read -p "$(echo -e '输入选项 [y/n]: ')" choice
        case $choice in
            [Yy]* )
                NEED_FINETUNE="y"
                echo -e "已选择: 是，将配置微调参数。\n"
                break
                ;;
            [Nn]* )
                NEED_FINETUNE="n"
                echo -e "已选择: 否，将跳过微调配置。\n"
                break
                ;;
            * )
                echo -e "无效选项，请输入 y 或 n"
                ;;
        esac
    done
}

# 询问是否使用 vLLM
ask_vllm_option() {
    echo -e "请问是否需要生成 vLLM 相关的推理脚本？"
    echo -e "(注: vLLM 推理速度更快，但通常需要更大的显存。若显存有限，建议选择 n 仅使用 PT 引擎)"
    while true; do
        read -p "$(echo -e '输入选项 [y/n]: ')" choice
        case $choice in
            [Yy]* )
                USE_VLLM="y"
                echo -e "已选择: 是，将生成 vLLM 相关脚本。\n"
                break
                ;;
            [Nn]* )
                USE_VLLM="n"
                echo -e "已选择: 否，将仅生成 PT (PyTorch/HuggingFace) 引擎脚本。\n"
                break
                ;;
            * )
                echo -e "无效选项，请输入 y 或 n"
                ;;
        esac
    done
}

# 配置微调参数
configure_finetune_params() {
    echo -e "\n--- 配置微调参数 ---"
    
    # 输出目录
    FINETUNE_OUTPUT_DIR="$TARGET_DIR/lora"
    echo -e "微调输出目录 (默认: $FINETUNE_OUTPUT_DIR):"
    read -p "$(echo -e '输入目录 (留空使用默认): ')" custom_output
    if [[ -n "$custom_output" ]]; then
        FINETUNE_OUTPUT_DIR="$custom_output"
    fi
    mkdir -p "$FINETUNE_OUTPUT_DIR"
    echo -e "微调输出目录: $FINETUNE_OUTPUT_DIR\n"
    
    # 数据集
    echo -e "请输入训练数据集 (格式: 'dataset_id#sample_count'，多个用空格分隔):"
    echo -e "参考: https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html"
    echo -e "示例: 'AI-ModelScope/alpaca-gpt4-data-zh#500 AI-ModelScope/alpaca-gpt4-data-en#500 swift/self-cognition#500'"
    read -p "$(echo -e '输入数据集: ')" DATASETS
    
    if [[ -z "$DATASETS" ]]; then
        DATASETS="AI-ModelScope/alpaca-gpt4-data-zh#500 AI-ModelScope/alpaca-gpt4-data-en#500 swift/self-cognition#500"
        echo -e "使用默认数据集: $DATASETS"
    fi
    echo
    
    # 批次大小
    read -p "$(echo -e '每设备训练批次大小 (默认: 1): ')" batch_size
    if [[ -n "$batch_size" && "$batch_size" =~ ^[0-9]+$ ]]; then
        BATCH_SIZE=$batch_size
    fi
    echo -e "每设备训练批次大小: $BATCH_SIZE\n"
    
    # 梯度累积步数
    read -p "$(echo -e '梯度累积步数 (默认: 16): ')" grad_accum
    if [[ -n "$grad_accum" && "$grad_accum" =~ ^[0-9]+$ ]]; then
        GRAD_ACCUM_STEPS=$grad_accum
    fi
    echo -e "梯度累积步数: $GRAD_ACCUM_STEPS\n"
    
    # LoRA rank
    read -p "$(echo -e 'LoRA rank (默认: 8): ')" lora_rank
    if [[ -n "$lora_rank" && "$lora_rank" =~ ^[0-9]+$ ]]; then
        LORA_RANK=$lora_rank
    fi
    echo -e "LoRA rank: $LORA_RANK\n"
    
    # LoRA alpha
    read -p "$(echo -e 'LoRA alpha (默认: 32): ')" lora_alpha
    if [[ -n "$lora_alpha" && "$lora_alpha" =~ ^[0-9]+$ ]]; then
        LORA_ALPHA=$lora_alpha
    fi
    echo -e "LoRA alpha: $LORA_ALPHA\n"
    
    # 训练轮数
    read -p "$(echo -e '训练轮数 (默认: 1): ')" num_epochs
    if [[ -n "$num_epochs" && "$num_epochs" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        NUM_EPOCHS=$num_epochs
    fi
    echo -e "训练轮数: $NUM_EPOCHS\n"
    
    # 最大长度
    read -p "$(echo -e '最大序列长度 (默认: 2048): ')" max_length
    if [[ -n "$max_length" && "$max_length" =~ ^[0-9]+$ ]]; then
        MAX_LENGTH=$max_length
    fi
    echo -e "最大序列长度: $MAX_LENGTH\n"
    
    # 目标模块
    read -p "$(echo -e '目标模块 (默认: all-linear): ')" target_modules
    if [[ -n "$target_modules" ]]; then
        TARGET_MODULES=$target_modules
    fi
    echo -e "目标模块: $TARGET_MODULES\n"
}

# 执行模型下载
download_model() {
    echo -e "开始下载模型..."
    echo -e "站点: $SELECTED_SITE"
    echo -e "模型ID: $MODEL_ID"
    
    # 创建目标目录
    TARGET_DIR="$SAVE_PATH/$MODEL_ID"
    mkdir -p "$TARGET_DIR"
    
    # 根据不同站点设置环境变量或命令
    case $SELECTED_SITE in
        "hf-mirror")
            echo -e "保存路径: $TARGET_DIR"
            
            # 设置 HF_ENDPOINT 环境变量
            export HF_ENDPOINT="https://hf-mirror.com"
            
            if ! command -v huggingface-cli &> /dev/null; then
                echo -e "正在安装 huggingface_hub..."
                pip install huggingface_hub --upgrade
            fi
            
            if [[ "$DOWNLOAD_OPTION" == "all" ]]; then
                huggingface-cli download \
                    --repo-type model \
                    --resume-download \
                    "$MODEL_ID" \
                    --local-dir "$TARGET_DIR" \
                    --local-dir-use-symlinks False
            else
                IFS=',' read -ra FILES <<< "$file_list"
                for file in "${FILES[@]}"; do
                    file=$(echo "$file" | xargs)
                    huggingface-cli download \
                        --repo-type model \
                        --filename "$file" \
                        "$MODEL_ID" \
                        --local-dir "$TARGET_DIR"
                done
            fi
            ;;
            
        "modelscope")
            echo -e "保存路径: $TARGET_DIR "
            
            # 检查是否安装了 modelscope
            if ! python3 -c "from modelscope.hub.snapshot_download import snapshot_download" &> /dev/null; then
                echo -e "正在安装 modelscope... "
                pip install modelscope --upgrade
            fi
            
            # 使用 modelscope 下载
            if [[ "$DOWNLOAD_OPTION" == "all" ]]; then
                python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
import os

# 确保父目录存在
os.makedirs('$TARGET_DIR', exist_ok=True)
snapshot_download(
    model_id='$MODEL_ID',
    revision='master',
    local_dir='$TARGET_DIR'
)
"
            else
                echo -e "ModelScope 不支持部分文件下载，将下载完整模型"
                python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
import os

os.makedirs('$TARGET_DIR', exist_ok=True)
snapshot_download(
    model_id='$MODEL_ID',
    revision='master',
    local_dir='$TARGET_DIR'
)
"
            fi
            ;;
    esac
    
    # 确保TARGET_DIR指向正确的路径
    if [[ "$SELECTED_SITE" == "modelscope" ]]; then
        TARGET_DIR="$SAVE_PATH/$MODEL_ID"
    fi
    
    if [[ $? -eq 0 ]]; then
        echo -e "✓ 模型下载成功！"
    else
        echo -e "✗ 模型下载失败"
        exit 1
    fi
}

# 生成推理脚本
generate_inference_scripts() {
    echo -e "\n正在生成推理脚本..."
    
    SCRIPTS_DIR="$TARGET_DIR/scripts"
    mkdir -p "$SCRIPTS_DIR"
    
    # 推理脚本文件名
    INFERENCE_PT_SCRIPT="$SCRIPTS_DIR/1-命令行推理-pt.sh"
    APP_PT_SCRIPT="$SCRIPTS_DIR/2-界面推理-pt.sh"
    DEPLOY_PT_SCRIPT="$SCRIPTS_DIR/3-API接口-pt.sh"
    
    INFERENCE_VLLM_SCRIPT="$SCRIPTS_DIR/1-命令行推理-vllm.sh"
    APP_VLLM_SCRIPT="$SCRIPTS_DIR/2-界面推理-vllm.sh"
    DEPLOY_VLLM_SCRIPT="$SCRIPTS_DIR/3-API接口-vllm.sh"
    
    UPGRADE_SCRIPT="/root/7-升级_ms_swift.sh"
    
    # 1. 生成PT引擎命令行推理脚本 (总是生成)
    cat > "$INFERENCE_PT_SCRIPT" << EOF
#!/bin/bash
# PT引擎命令行推理脚本
# 模型: $MODEL_ID
# 模型类型: $MODEL_TYPE

CUDA_VISIBLE_DEVICES=0 swift infer \\
    --model "$TARGET_DIR" \\
    --model_type "$MODEL_TYPE" \\
    --stream true \\
    --infer_backend pt \\
    --max_new_tokens 2048
EOF

    # 2. 生成PT引擎Web界面推理脚本 (总是生成)
    cat > "$APP_PT_SCRIPT" << EOF
#!/bin/bash
# PT引擎Web界面推理脚本
# 模型: $MODEL_ID
# 模型类型: $MODEL_TYPE

swift app --model '$TARGET_DIR' --model_type '$MODEL_TYPE' --studio_title '$MODEL_NAME' --lang zh --max_new_tokens 2048 --infer_backend pt --server_port 6006
EOF

    # 3. 生成PT引擎API部署脚本 (总是生成)
    cat > "$DEPLOY_PT_SCRIPT" << EOF
#!/bin/bash
# PT引擎API服务部署脚本
# 模型: $MODEL_ID
# 模型类型: $MODEL_TYPE

CUDA_VISIBLE_DEVICES=0 swift deploy \\
    --model "$TARGET_DIR" \\
    --model_type "$MODEL_TYPE" \\
    --infer_backend pt \\
    --served_model_name $MODEL_NAME \\
    --port 6008
EOF

    # 4. 根据用户选择生成 VLLM 脚本
    if [[ "$USE_VLLM" == "y" || "$USE_VLLM" == "Y" ]]; then
        cat > "$INFERENCE_VLLM_SCRIPT" << EOF
#!/bin/bash
# VLLM引擎命令行推理脚本
# 模型: $MODEL_ID
# 模型类型: $MODEL_TYPE

CUDA_VISIBLE_DEVICES=0 swift infer \\
    --model "$TARGET_DIR" \\
    --model_type "$MODEL_TYPE" \\
    --stream true \\
    --infer_backend vllm \\
    --max_new_tokens 2048 \\
    --vllm_max_model_len 4096
EOF

        cat > "$APP_VLLM_SCRIPT" << EOF
#!/bin/bash
# VLLM引擎Web界面推理脚本
# 模型: $MODEL_ID
# 模型类型: $MODEL_TYPE

swift app --model '$TARGET_DIR' --model_type '$MODEL_TYPE' --studio_title '$MODEL_NAME' --lang zh --max_new_tokens 2048 --infer_backend vllm --vllm_max_model_len 4096 --server_port 6006
EOF

        cat > "$DEPLOY_VLLM_SCRIPT" << EOF
#!/bin/bash
# VLLM引擎API服务部署脚本
# 模型: $MODEL_ID
# 模型类型: $MODEL_TYPE

CUDA_VISIBLE_DEVICES=0 swift deploy \\
    --model "$TARGET_DIR" \\
    --model_type "$MODEL_TYPE" \\
    --infer_backend vllm \\
    --served_model_name $MODEL_NAME \\
    --vllm_max_model_len 4096 \\
    --port 6008
EOF
        chmod +x "$INFERENCE_VLLM_SCRIPT"
        chmod +x "$APP_VLLM_SCRIPT"
        chmod +x "$DEPLOY_VLLM_SCRIPT"
        echo -e "  ✓ vLLM 推理脚本生成成功"
    else
        echo -e "  ⚠ 已跳过 vLLM 脚本生成"
    fi

    # 5. 生成ms-swift升级脚本 (总是生成)
    cat > "$UPGRADE_SCRIPT" << 'EOF'
#!/bin/bash
# 升级ms-swift脚本

echo "正在从GitHub仓库升级ms-swift..."

# 检查是否已经有克隆的仓库
if [ -d "ms-swift" ]; then
    cd ms-swift
    echo "拉取最新代码..."
    git pull origin main
    cd ..
else
    echo "克隆ms-swift仓库..."
    git clone https://github.com/modelscope/ms-swift.git
fi

echo "安装/升级依赖..."
cd ms-swift
pip install -e .
cd ..

echo "ms-swift升级完成！"
EOF

    # 赋予执行权限 (PT 脚本和升级脚本)
    chmod +x "$INFERENCE_PT_SCRIPT"
    chmod +x "$APP_PT_SCRIPT"
    chmod +x "$DEPLOY_PT_SCRIPT"
    chmod +x "$UPGRADE_SCRIPT"
    
    echo -e "✓ 基础推理脚本生成成功！"
}

# 生成微调脚本
generate_finetune_scripts() {
    echo -e "\n正在生成微调脚本..."
    
    SCRIPTS_DIR="$TARGET_DIR/scripts"
    mkdir -p "$SCRIPTS_DIR"
    
    # 微调脚本文件名
    FINETUNE_SCRIPT="$SCRIPTS_DIR/4-微调.sh"
    INFER_AFTER_FINETUNE_SCRIPT="$SCRIPTS_DIR/5-微调后推理-需修改路径.sh"
    MERGE_FINETUNE_SCRIPT="$SCRIPTS_DIR/6-合并微调-需修改路径.sh"
    
    # 生成微调脚本
    cat > "$FINETUNE_SCRIPT" << EOF
#!/bin/bash
# 微调脚本
# 模型: $MODEL_ID
# 模型类型: $MODEL_TYPE

CUDA_VISIBLE_DEVICES=0 \\
swift sft \\
    --model "$TARGET_DIR" \\
    --model_type "$MODEL_TYPE" \\
    --train_type lora \\
    --dataset $DATASETS \\
    --torch_dtype bfloat16 \\
    --num_train_epochs $NUM_EPOCHS \\
    --per_device_train_batch_size $BATCH_SIZE \\
    --per_device_eval_batch_size $BATCH_SIZE \\
    --learning_rate 1e-4 \\
    --lora_rank $LORA_RANK \\
    --lora_alpha $LORA_ALPHA \\
    --target_modules $TARGET_MODULES \\
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \\
    --eval_steps 50 \\
    --save_steps 50 \\
    --save_total_limit 2 \\
    --logging_steps 5 \\
    --max_length $MAX_LENGTH \\
    --output_dir "$FINETUNE_OUTPUT_DIR" \\
    --system 'You are a helpful assistant.' \\
    --warmup_ratio 0.05 \\
    --dataloader_num_workers 4 \\
    --model_author swift \\
    --model_name swift-robot
EOF

    # 生成微调后推理参考脚本 (修复了原脚本单引号EOF导致变量不展开的问题)
    cat > "$INFER_AFTER_FINETUNE_SCRIPT" << EOF
#!/bin/bash
# 微调后推理参考脚本

echo "请修改 /path/to/fine-tuned-lora 为实际的路径"

# 修改 /path/to/fine-tuned-lora 为实际的路径
# 示例路径: /root/lora/v0-20251010-101010/checkpoint-95
CUDA_VISIBLE_DEVICES=0 \\
swift infer \\
    --adapters /path/to/fine-tuned-lora \\
    --model_type "$MODEL_TYPE" \\
    --stream true \\
    --temperature 0 \\
    --max_new_tokens 2048
EOF

    # 生成合并微调脚本
    cat > "$MERGE_FINETUNE_SCRIPT" << EOF
#!/bin/bash
# 合并微调脚本

echo "请修改 /path/to/fine-tuned-lora 为实际的路径"
echo "示例路径：  /root/big-models/Qwen/Qwen2.5-7B-Instruct/lora/v0-20251020-101010/checkpoint-94"
echo "---------------------"
echo "合并后模型路径，请查看输出文本。"
echo "合并示例提示（以实际为准）： [INFO:swift] Successfully merged LoRA and saved in /root/big-models/Qwen/Qwen2.5-7B-Instruct/lora/v0-20251010-101010/checkpoint-94-merged."
swift export \\
    --adapters /path/to/fine-tuned-lora \\
    --merge_lora true
EOF

    # 赋予执行权限
    chmod +x "$FINETUNE_SCRIPT"
    chmod +x "$INFER_AFTER_FINETUNE_SCRIPT"
    chmod +x "$MERGE_FINETUNE_SCRIPT"
    
    echo -e "✓ 微调脚本生成成功！"
    echo -e "脚本位置: $SCRIPTS_DIR"
    echo -e "包含以下脚本:"
    echo -e "  • $FINETUNE_SCRIPT (微调脚本)"
    echo -e "  • $INFER_AFTER_FINETUNE_SCRIPT (微调后推理参考脚本)"
    echo -e "  • $MERGE_FINETUNE_SCRIPT (合并微调脚本)"
}

# 显示完成信息
show_completion() {
    cat << EOF

🎉 操作执行完成！

模型已成功下载并配置完毕。

您可以使用以下命令进行操作：

1. PT引擎命令行推理:
   $ chmod +x $TARGET_DIR/scripts/1-命令行推理-pt.sh
   $ $TARGET_DIR/scripts/1-命令行推理-pt.sh

EOF

    if [[ "$USE_VLLM" == "y" || "$USE_VLLM" == "Y" ]]; then
        cat << EOF
2. VLLM引擎命令行推理:
   $ chmod +x $TARGET_DIR/scripts/1-命令行推理-vllm.sh
   $ $TARGET_DIR/scripts/1-命令行推理-vllm.sh

3. PT引擎Web界面:
   $ chmod +x $TARGET_DIR/scripts/2-界面推理-pt.sh
   $ $TARGET_DIR/scripts/2-界面推理-pt.sh

4. VLLM引擎Web界面:
   $ chmod +x $TARGET_DIR/scripts/2-界面推理-vllm.sh
   $ $TARGET_DIR/scripts/2-界面推理-vllm.sh

5. PT引擎API服务:
   $ chmod +x $TARGET_DIR/scripts/3-API接口-pt.sh
   $ $TARGET_DIR/scripts/3-API接口-pt.sh

6. VLLM引擎API服务:
   $ chmod +x $TARGET_DIR/scripts/3-API接口-vllm.sh
   $ $TARGET_DIR/scripts/3-API接口-vllm.sh
EOF
    else
        cat << EOF
2. PT引擎Web界面:
   $ chmod +x $TARGET_DIR/scripts/2-界面推理-pt.sh
   $ $TARGET_DIR/scripts/2-界面推理-pt.sh

3. PT引擎API服务:
   $ chmod +x $TARGET_DIR/scripts/3-API接口-pt.sh
   $ $TARGET_DIR/scripts/3-API接口-pt.sh

   (注: 您选择了不生成 vLLM 脚本。如需使用 vLLM，请重新运行本工具并选择 y)
EOF
    fi

    if [[ "$NEED_FINETUNE" == "y" || "$NEED_FINETUNE" == "Y" ]]; then
        cat << EOF

7. 微调模型:
   $ chmod +x $TARGET_DIR/scripts/4-微调.sh
   $ $TARGET_DIR/scripts/4-微调.sh

8. 微调后推理:
   $ # 首先编辑 $TARGET_DIR/scripts/5-微调后推理-需修改路径.sh
   $ # 将 /path/to/fine-tuned-lora 替换为实际的微调检查点路径
   $ chmod +x $TARGET_DIR/scripts/5-微调后推理-需修改路径.sh
   $ $TARGET_DIR/scripts/5-微调后推理-需修改路径.sh

9. 合并微调结果:
   $ # 首先编辑 $TARGET_DIR/scripts/6-合并微调-需修改路径.sh
   $ # 将 /path/to/fine-tuned-lora 替换为实际的微调检查点路径
   $ chmod +x $TARGET_DIR/scripts/6-合并微调-需修改路径.sh
   $ $TARGET_DIR/scripts/6-合并微调-需修改路径.sh
EOF
    else
        cat << EOF

(注: 您选择了跳过微调配置。如需微调，请重新运行本工具并选择 y)
EOF
    fi

    cat << EOF

10. 升级 ms-swift:
   $ chmod +x /root/7-升级_ms_swift.sh
   $ /root/7-升级_ms_swift.sh

感谢使用一锤定音大模型工具 (CHUIZI，公众号：锤子代码)！
EOF
}

# 主函数
main() {
    show_welcome
    check_dependencies
    select_download_site
    get_model_id
    get_model_type
    select_save_path
    select_download_option
    download_model
    
    # 下载完成后，询问进阶选项
    ask_finetune_option
    ask_vllm_option
    
    # 生成脚本
    generate_inference_scripts
    
    # 根据用户选择，决定是否生成微调脚本
    if [[ "$NEED_FINETUNE" == "y" || "$NEED_FINETUNE" == "Y" ]]; then
        configure_finetune_params
        generate_finetune_scripts
    fi
    
    show_completion
}

# 运行主程序
main "$@"
