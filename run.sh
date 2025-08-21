#!/bin/bash

# 显示运行时脚本目录并进入
echo "运行目录: ${RUNTIME_SCRIPT_DIR}"
cd ${RUNTIME_SCRIPT_DIR}

# 设置环境变量（多模态特征ID，默认81）
MM_EMB_ID=${MM_EMB_ID:-81}
export MM_EMB_ID

# 定义输出目录（平台指定或默认）
OUTPUT_DIR=${TRAIN_OUTPUT_PATH:-./output}
mkdir -p "$OUTPUT_DIR"

# RQVAE核心参数（与模型代码保持一致）
INPUT_DIM=${INPUT_DIM:-32}  # 81=32, 82=1024, 83=3584...需根据实际特征调整
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-"512 256"}
LATENT_DIM=${LATENT_DIM:-128}
NUM_CODEBOOKS=${NUM_CODEBOOKS:-4}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-"128 128 128 128"}
KMEANS_METHOD=${KMEANS_METHOD:-"kmeans"}
DISTANCES_METHOD=${DISTANCES_METHOD:-"l2"}
RQVAE_EPOCHS=${RQVAE_EPOCHS:-10}
RQVAE_LR=${RQVAE_LR:-0.0005}

# 训练模式（separate=两阶段，joint=联合训练）
TRAINING_MODE=${TRAINING_MODE:-"separate"}

if [ "$TRAINING_MODE" = "separate" ]; then
    echo "===== 开始两阶段训练 ====="
    
    # 第一阶段：训练RQVAE并生成语义ID
    echo "===== 阶段1/2：训练RQVAE ====="
    python -u rqvae_pretrain.py \
        --data_dir ./data \  # 平台数据根目录（根据实际情况调整）
        --feature_id $MM_EMB_ID \
        --input_dim $INPUT_DIM \
        --hidden_channels $HIDDEN_CHANNELS \
        --latent_dim $LATENT_DIM \
        --num_codebooks $NUM_CODEBOOKS \
        --codebook_size $CODEBOOK_SIZE \
        --kmeans_method $KMEANS_METHOD \
        --distances_method $DISTANCES_METHOD \
        --lr $RQVAE_LR \
        --num_epochs $RQVAE_EPOCHS \
        --batch_size 256 \
        --num_workers 0 \
        --device cuda \
        --save_dir "$OUTPUT_DIR/rqvae"  # RQVAE模型和语义ID保存目录
    
    # 检查RQVAE训练是否成功
    if [ $? -ne 0 ]; then
        echo "RQVAE训练失败，退出脚本"
        exit 1
    fi
    
    # 确认语义ID文件存在
    SEMANTIC_ID_FILE="$OUTPUT_DIR/rqvae/semantic_ids_${MM_EMB_ID}.pkl"
    if [ ! -f "$SEMANTIC_ID_FILE" ]; then
        echo "错误：未找到语义ID文件 $SEMANTIC_ID_FILE"
        exit 1
    fi
    echo "已找到语义ID文件：$SEMANTIC_ID_FILE"
    
    # 第二阶段：用语义ID训练Baseline模型
    echo "===== 阶段2/2：训练Baseline模型 ====="
    python -u main.py \
        --data_dir ./data \  # 平台数据根目录（根据实际情况调整）
        --mm_emb_id $MM_EMB_ID \
        --use_semantic_id \  # 启用语义ID特征
        --semantic_id_path "$SEMANTIC_ID_FILE" \
        --num_codebooks $NUM_CODEBOOKS \  # 传递码本数量，与RQVAE保持一致
        --output_dir "$OUTPUT_DIR/baseline" \
        --maxlen 50 \  # 根据实际需求调整
        --hidden_units 512 \
        --num_blocks 2 \
        --num_heads 4 \
        --dropout_rate 0.1 \
        --batch_size 128 \
        --epochs 10 \
        --device cuda
    
else
    # 联合训练模式（复用原有逻辑，确保参数兼容）
    echo "===== 开始联合训练模式 ====="
    
    RQVAE_MODEL_PATH=${RQVAE_MODEL_PATH:-""}
    RQVAE_MODEL_PARAM=""
    if [ -n "$RQVAE_MODEL_PATH" ] && [ -f "$RQVAE_MODEL_PATH" ]; then
        echo "使用预训练RQVAE模型: $RQVAE_MODEL_PATH"
        RQVAE_MODEL_PARAM="--rqvae_state_dict_path $RQVAE_MODEL_PATH"
    fi
    
    python -u main.py \
        --data_dir ./data \
        --mm_emb_id $MM_EMB_ID \
        --use_rqvae \
        --rqvae_loss_weight 0.1 \
        --input_dim $INPUT_DIM \
        --hidden_channels $HIDDEN_CHANNELS \
        --latent_dim $LATENT_DIM \
        --num_codebooks $NUM_CODEBOOKS \
        --codebook_size $CODEBOOK_SIZE \
        --output_dir "$OUTPUT_DIR/joint" \
        --maxlen 50 \
        --hidden_units 512 \
        --epochs 10 \
        $RQVAE_MODEL_PARAM
fi

# 检查训练最终状态
if [ $? -ne 0 ]; then
    echo "训练失败"
    exit 1
fi

echo "所有训练完成，输出目录：$OUTPUT_DIR"
