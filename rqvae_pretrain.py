import argparse
import os
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 假设你已有的RQVAE模型定义
from model_rqvae import RQVAE  # 替换为你的RQVAE模型路径
from dataset import MultiModalEmbDataset  # 加载多模态特征的数据集


def get_args():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_path', required=True, type=str, help='多模态特征数据路径')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    
    # RQVAE参数（与你的model.py对齐）
    parser.add_argument('--input_dim', default=1024, type=int)
    parser.add_argument('--hidden_channels', nargs='+', default=[512, 256], type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--num_codebooks', default=4, type=int)
    parser.add_argument('--codebook_size', nargs='+', default=[128, 128, 128, 128], type=int)
    parser.add_argument('--shared_codebook', action='store_true')
    
    # 训练参数
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--beta', default=0.25, type=float, help='量化损失权重')
    parser.add_argument('--gamma', default=0.1, type=float, help='对比损失权重')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--save_dir', default='./rqvae_pretrain', type=str)
    
    # 输出语义ID路径
    parser.add_argument('--semantic_id_save_path', default='./semantic_ids.pkl', type=str)
    return parser.parse_args()


# 对比损失：增强语义ID的区分度
def contrastive_loss(semantic_embs, item_labels):
    """
    semantic_embs: 语义ID对应的嵌入 [batch_size, sem_dim]
    item_labels: item的真实类别（用于判断同类）[batch_size]
    """
    # 计算余弦相似度
    sim = F.cosine_similarity(semantic_embs.unsqueeze(1), semantic_embs.unsqueeze(0), dim=2)  # [B, B]
    # 同类掩码（对角线为0，避免自相似）
    mask = (item_labels.unsqueeze(1) == item_labels.unsqueeze(0)).float() - torch.eye(len(item_labels), device=semantic_embs.device)
    # 正样本对相似度之和 / 所有样本对相似度之和
    pos_sim_sum = (sim * mask).sum(1)
    total_sim_sum = sim.sum(1) - sim.diag()  # 减去自身相似度
    # 对比损失（让同类相似度更高）
    return -torch.log((pos_sim_sum + 1e-8) / (total_sim_sum + 1e-8)).mean()


def main():
    args = get_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # 日志和TensorBoard
    writer = SummaryWriter(os.path.join(args.save_dir, 'tb_logs'))
    log_file = open(os.path.join(args.save_dir, 'train.log'), 'w')
    
    # 1. 加载多模态特征数据集（仅包含item的多模态嵌入和类别标签）
    dataset = MultiModalEmbDataset(args.data_path)  # 需实现：返回(item_id, mm_emb, label)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 2. 初始化RQVAE模型
    model = RQVAE(
        input_dim=args.input_dim,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        shared_codebook=args.shared_codebook
    ).to(args.device)
    
    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. 预训练RQVAE
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        recon_losses = 0.0
        quant_losses = 0.0
        contrast_losses = 0.0
        
        for batch in tqdm(dataloader, desc=f"RQVAE Epoch {epoch+1}"):
            item_ids, mm_embs, labels = batch  # mm_embs: [B, input_dim]
            mm_embs = mm_embs.to(args.device).float()
            labels = labels.to(args.device)
            
            # 前向传播：获取重构结果和量化损失
            recon, quant_loss, semantic_embs = model(mm_embs)  # 假设model返回(重构特征, 量化损失, 语义嵌入)
            
            # 计算重构损失（MSE）
            recon_loss = F.mse_loss(recon, mm_embs)
            
            # 计算对比损失（增强语义区分度）
            contrast_loss = contrastive_loss(semantic_embs, labels) if args.gamma > 0 else 0.0
            
            # 总损失
            loss = recon_loss + args.beta * quant_loss + args.gamma * contrast_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            recon_losses += recon_loss.item()
            quant_losses += quant_loss.item()
            if args.gamma > 0:
                contrast_losses += contrast_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_recon = recon_losses / len(dataloader)
        avg_quant = quant_losses / len(dataloader)
        avg_contrast = contrast_losses / len(dataloader) if args.gamma > 0 else 0.0
        
        # 日志记录
        log_msg = {
            'epoch': epoch+1,
            'total_loss': avg_loss,
            'recon_loss': avg_recon,
            'quant_loss': avg_quant,
            'contrast_loss': avg_contrast,
            'time': time.time()
        }
        print(log_msg)
        log_file.write(str(log_msg) + '\n')
        log_file.flush()
        
        # TensorBoard
        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/recon', avg_recon, epoch)
        writer.add_scalar('Loss/quant', avg_quant, epoch)
        if args.gamma > 0:
            writer.add_scalar('Loss/contrast', avg_contrast, epoch)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_rqvae.pt'))
    
    # 5. 生成并保存全量item的语义ID
    print("生成语义ID...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_rqvae.pt')))
    model.eval()
    semantic_id_dict = {}  # {item_id: [code1, code2, ...]}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="生成语义ID"):
            item_ids, mm_embs, _ = batch
            mm_embs = mm_embs.to(args.device).float()
            # 获取语义ID（码本索引）
            semantic_ids = model.get_codebook_indices(mm_embs)  # 假设返回[B, num_codebooks]的整数张量
            # 保存到字典
            for item_id, sid in zip(item_ids, semantic_ids.cpu().numpy().tolist()):
                semantic_id_dict[item_id] = sid
    
    # 保存语义ID
    with open(args.semantic_id_save_path, 'wb') as f:
        pickle.dump(semantic_id_dict, f)
    print(f"语义ID已保存至：{args.semantic_id_save_path}，共{len(semantic_id_dict)}个item")
    
    writer.close()
    log_file.close()


if __name__ == '__main__':
    main()
