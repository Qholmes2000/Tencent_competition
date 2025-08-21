import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset  # 需使用修改后的支持语义ID的Dataset
from model import BaselineModel  # 需使用支持语义ID的BaselineModel


def calculate_metrics(pos_logits, neg_logits, k=10):
    """计算推荐指标HR@k和NDCG@k"""
    metrics = {'hr': [], 'ndcg': []}
    for p_score, n_scores in zip(pos_logits.cpu().numpy(), neg_logits.cpu().numpy()):
        # 合并正负样本分数并排序
        all_scores = np.concatenate([[p_score], n_scores])
        sorted_indices = np.argsort(-all_scores)  # 降序排序
        pos_rank = np.where(sorted_indices == 0)[0][0] + 1  # 正样本排名（1-based）
        
        # HR@k：正样本是否在前k名
        metrics['hr'].append(1.0 if pos_rank <= k else 0.0)
        # NDCG@k：归一化折损累积增益
        dcg = 1.0 / np.log2(pos_rank + 1) if pos_rank <= k else 0.0
        idcg = 1.0 / np.log2(2)  # 理想情况：正样本排第1
        metrics['ndcg'].append(dcg / idcg)
    
    return {k: np.mean(v) for k, v in metrics.items()}


def get_args():
    parser = argparse.ArgumentParser()

    # 训练基础参数
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)  # 第二阶段训练轮次
    parser.add_argument('--num_workers', default=0, type=int)  # 数据加载进程数

    # 模型结构参数
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0001, type=float)  # L2正则系数
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)  # 可选：预训练模型路径
    parser.add_argument('--norm_first', action='store_true')

    # 多模态与语义ID参数（第二阶段核心）
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--use_semantic_id', action='store_true', help='启用语义ID特征（第二阶段必选）')
    parser.add_argument('--semantic_id_path', required=True, type=str, help='第一阶段生成的语义ID文件路径')
    parser.add_argument('--num_codebooks', default=4, type=int, help='语义ID码本数量（与RQVAE一致）')

    return parser.parse_args()


if __name__ == '__main__':
    # 初始化路径（从环境变量或默认值）
    TRAIN_LOG_PATH = os.environ.get('TRAIN_LOG_PATH', './logs')
    TRAIN_TF_EVENTS_PATH = os.environ.get('TRAIN_TF_EVENTS_PATH', './tf_events')
    TRAIN_DATA_PATH = os.environ.get('TRAIN_DATA_PATH', './data')
    TRAIN_CKPT_PATH = os.environ.get('TRAIN_CKPT_PATH', './checkpoints')
    
    # 创建目录
    Path(TRAIN_LOG_PATH).mkdir(parents=True, exist_ok=True)
    Path(TRAIN_TF_EVENTS_PATH).mkdir(parents=True, exist_ok=True)
    Path(TRAIN_CKPT_PATH).mkdir(parents=True, exist_ok=True)
    
    # 日志配置
    log_file = open(Path(TRAIN_LOG_PATH, 'train_stage2.log'), 'w')  # 第二阶段日志
    writer = SummaryWriter(TRAIN_TF_EVENTS_PATH)

    # 解析参数
    args = get_args()
    print(f"第二阶段训练参数: {args}")

    # 初始化数据集（加载语义ID）
    dataset = MyDataset(
        data_dir=TRAIN_DATA_PATH,
        args=args,
        semantic_id_path=args.semantic_id_path  # 传入第一阶段生成的语义ID
    )
    # 划分训练集和验证集
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    # 数据加载器（多进程加速）
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True  # 加速GPU传输
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )

    # 数据集元信息
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    print(f"数据集信息: 用户数={usernum}, 物品数={itemnum}")

    # 初始化模型（支持语义ID特征）
    model = BaselineModel(
        usernum=usernum,
        itemnum=itemnum,
        feat_statistics=feat_statistics,
        feat_types=feat_types,
        args=args
    ).to(args.device)

    # 参数初始化
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass  # 跳过无法用xavier初始化的参数
    
    # 零向量初始化（padding位置）
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # 加载预训练模型（可选）
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"加载预训练模型成功，从epoch {epoch_start_idx} 开始")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            raise RuntimeError("请检查模型路径")

    # 优化器与损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.l2_emb  # 全局L2正则
    )
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # 指标跟踪
    best_val_ndcg = -1.0
    best_val_hr = -1.0
    global_step = 0
    t0 = time.time()
    print("开始第二阶段训练（使用语义ID特征）")

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        # -------------------------- 训练阶段 --------------------------
        model.train()
        if args.inference_only:
            break
        
        train_loss = 0.0
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} 训练"):
            # 解析batch（包含语义ID特征）
            (seq, pos, neg, token_type, next_token_type, next_action_type,
             seq_feat, pos_feat, neg_feat, seq_sem_ids, pos_sem_ids, neg_sem_ids) = batch
            
            # 数据移至设备
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            seq_sem_ids = seq_sem_ids.to(args.device)  # 序列语义ID [B, maxlen, num_codebooks]
            pos_sem_ids = pos_sem_ids.to(args.device)  # 正样本语义ID
            neg_sem_ids = neg_sem_ids.to(args.device)  # 负样本语义ID

            # 模型前向传播（传入语义ID）
            pos_logits, neg_logits, _ = model(
                seq, pos, neg,
                token_type, next_token_type, next_action_type,
                seq_feat, pos_feat, neg_feat,
                seq_sem_ids, pos_sem_ids, neg_sem_ids  # 新增语义ID参数
            )

            # 计算损失
            indices = np.where(next_token_type == 1)  # 有效样本索引
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 日志记录
            train_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            if (step + 1) % 100 == 0:  # 每100步打印一次
                log_json = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'time': time.time()
                }
                log_file.write(json.dumps(log_json) + '\n')
                log_file.flush()
                print(log_json)
            
            global_step += 1
        
        #  epoch级训练日志
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} 训练平均损失: {avg_train_loss:.4f}")

        # -------------------------- 验证阶段 --------------------------
        model.eval()
        valid_loss = 0.0
        val_metrics = {'hr': [], 'ndcg': []}
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Epoch {epoch} 验证"):
                # 解析batch（同训练）
                (seq, pos, neg, token_type, next_token_type, next_action_type,
                 seq_feat, pos_feat, neg_feat, seq_sem_ids, pos_sem_ids, neg_sem_ids) = batch
                
                # 数据移至设备
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                seq_sem_ids = seq_sem_ids.to(args.device)
                pos_sem_ids = pos_sem_ids.to(args.device)
                neg_sem_ids = neg_sem_ids.to(args.device)

                # 模型前向传播
                pos_logits, neg_logits, _ = model(
                    seq, pos, neg,
                    token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat,
                    seq_sem_ids, pos_sem_ids, neg_sem_ids
                )

                # 验证损失
                indices = np.where(next_token_type == 1)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                valid_loss += loss.item()

                # 计算推荐指标（HR@10, NDCG@10）
                metrics = calculate_metrics(pos_logits[indices], neg_logits[indices], k=10)
                val_metrics['hr'].append(metrics['hr'])
                val_metrics['ndcg'].append(metrics['ndcg'])
        
        # 验证指标汇总
        avg_valid_loss = valid_loss / len(valid_loader)
        avg_val_hr = np.mean(val_metrics['hr'])
        avg_val_ndcg = np.mean(val_metrics['ndcg'])
        
        # 日志记录
        writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
        writer.add_scalar('Metrics/val_hr@10', avg_val_hr, epoch)
        writer.add_scalar('Metrics/val_ndcg@10', avg_val_ndcg, epoch)
        
        val_log = {
            'epoch': epoch,
            'valid_loss': avg_valid_loss,
            'val_hr@10': avg_val_hr,
            'val_ndcg@10': avg_val_ndcg,
            'time': time.time()
        }
        log_file.write(json.dumps(val_log) + '\n')
        log_file.flush()
        print(f"Epoch {epoch} 验证结果: {val_log}")

        # 保存最佳模型（基于NDCG@10）
        if avg_val_ndcg > best_val_ndcg or (avg_val_ndcg == best_val_ndcg and avg_val_hr > best_val_hr):
            best_val_ndcg = avg_val_ndcg
            best_val_hr = avg_val_hr
            best_model_path = Path(TRAIN_CKPT_PATH, f"stage2_epoch={epoch}_ndcg={best_val_ndcg:.4f}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"更新最佳模型: {best_model_path}")

    # 训练结束
    print("第二阶段训练完成")
    writer.close()
    log_file.close()
