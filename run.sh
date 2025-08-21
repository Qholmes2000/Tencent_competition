#!/bin/bash

# 进入脚本目录，失败则退出
cd ${RUNTIME_SCRIPT_DIR} || { echo "无法进入脚本目录"; exit 1; }

# 环境变量与参数配置
MM_EMB_ID=${MM_EMB_ID:-81}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/data_ams/training_data}  # 平台提供的训练数据路径
OUTPUT_DIR=${TRAIN_OUTPUT_PATH:-./output}
mkdir -p "$OUTPUT_DIR" || { echo "无法创建输出目录"; exit 1; }

# 训练模式（默认两阶段）
TRAINING_MODE=${TRAINING_MODE:-"separate"}

# 第一阶段：RQVAE训练（仅两阶段模式）
if [ "$TRAINING_MODE" = "separate" ]; then
    echo "===== 阶段1/2：训练RQVAE ====="
    python -u rqvae_pretrain.py \
        --data_dir "${TRAIN_DATA_PATH}" \
        --feature_id "${MM_EMB_ID}" \
        --input_dim 32 \
        --hidden_channels 512 256 \
        --latent_dim 128 \
        --num_codebooks 4 \
        --codebook_size 128 128 128 128 \
        --kmeans_method "kmeans" \
        --distances_method "l2" \
        --lr 0.0005 \
        --num_epochs 10 \
        --batch_size 256 \
        --num_workers 8 \
        --device "cuda" \
        --save_dir "${OUTPUT_DIR}/rqvae"
    # 检查RQVAE是否成功，失败则退出
    if [ $? -ne 0 ]; then
        echo "阶段1失败：RQVAE训练出错"
        exit 1
    fi

    # 检查语义ID文件
    SEMANTIC_ID_FILE="${OUTPUT_DIR}/rqvae/semantic_ids_${MM_EMB_ID}.pkl"
    if [ ! -f "$SEMANTIC_ID_FILE" ]; then
        echo "阶段1失败：未生成语义ID文件 ${SEMANTIC_ID_FILE}"
        exit 1
    fi

    # 第二阶段：训练Baseline（使用语义ID）
    echo "===== 阶段2/2：训练Baseline ====="
    python -u main.py \
        --data_dir "${TRAIN_DATA_PATH}" \
        --mm_emb_id "${MM_EMB_ID}" \
        --use_semantic_id \
        --semantic_id_path "${SEMANTIC_ID_FILE}" \
        --num_codebooks 4 \
        --batch_size 128 \
        --num_epochs 10 \
        --device "cuda" \
        --num_workers 8
    # 检查Baseline训练是否成功
    if [ $? -ne 0 ]; then
        echo "阶段2失败：Baseline训练出错"
        exit 1
    fi

else
    # 联合训练模式（不使用语义ID，避免参数缺失）
    echo "===== 联合训练模式 ====="
    python -u main.py \
        --data_dir "${TRAIN_DATA_PATH}" \
        --mm_emb_id "${MM_EMB_ID}" \
        --use_rqvae \
        --batch_size 128 \
        --num_epochs 10 \
        --device "cuda" \
        --num_workers 8
    # 检查联合训练是否成功
    if [ $? -ne 0 ]; then
        echo "联合训练失败"
        exit 1
    fi
fi

# 只有所有阶段成功，才打印训练完成
echo "所有训练完成，输出目录：${OUTPUT_DIR}"
