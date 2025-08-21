import argparse
import os
import pickle
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import MmEmbDataset, RQVAE


def get_args():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_dir', required=True, type=str, help='数据根目录（包含creative_emb）')
    parser.add_argument('--feature_id', default='81', type=str, help='多模态特征ID（81-86）')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    
    # RQVAE参数
    parser.add_argument('--input_dim', default=1024, type=int, help='多模态emb维度（81为32，82为1024等）')
    parser.add_argument('--hidden_channels', nargs='+', default=[512, 256], type=int, help='编码器/解码器隐藏层')
    parser.add_argument('--latent_dim', default=128, type=int, help='潜在空间维度')
    parser.add_argument('--num_codebooks', default=4, type=int, help='码本数量')
    parser.add_argument('--codebook_size', nargs='+', default=[128, 128, 128, 128], type=int, help='每个码本大小')
    parser.add_argument('--shared_codebook', action='store_true', help='是否共享码本')
    parser.add_argument('--kmeans_method', default='kmeans', choices=['kmeans', 'bkmeans'], help='聚类方法')
    parser.add_argument('--kmeans_iters', default=100, type=int, help='kmeans迭代次数')
    parser.add_argument('--distances_method', default='l2', choices=['l2', 'cosine'], help='距离计算方式')
    parser.add_argument('--loss_beta', default=0.25, type=float, help='RQ损失权重')
    
    # 训练参数
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--save_dir', default='./rqvae_pretrain', type=str, help='模型和语义ID保存目录')
    
    return parser.parse_args()


def main():
    args = get_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志配置
    writer = SummaryWriter(save_dir / 'tb_logs')
    log_file = open(save_dir / 'train.log', 'w')
    
    # 1. 加载多模态数据集
    dataset = MmEmbDataset(
        data_dir=args.data_dir,
        feature_id=args.feature_id
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    print(f"数据集加载完成：共{len(dataset)}个item，特征ID={args.feature_id}")
    
    # 2. 初始化RQVAE模型
    model = RQVAE(
        input_dim=args.input_dim,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        shared_codebook=args.shared_codebook,
        kmeans_method=args.kmeans_method,
        kmeans_iters=args.kmeans_iters,
        distances_method=args.distances_method,
        loss_beta=args.loss_beta,
        device=args.device
    ).to(args.device)
    
    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. 预训练RQVAE
    best_total_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        epoch_start = time.time()
        total_losses = 0.0
        recon_losses = 0.0
        rq_losses = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            tid_batch, emb_batch = batch  # emb_batch: [B, input_dim]
            emb_batch = emb_batch.to(args.device)
            
            # 前向传播
            x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss = model(emb_batch)
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 累计损失
            total_losses += total_loss.item()
            recon_losses += recon_loss.item()
            rq_losses += rqvae_loss.item()
        
        # 计算平均损失
        avg_total = total_losses / len(dataloader)
        avg_recon = recon_losses / len(dataloader)
        avg_rq = rq_losses / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        # 日志记录
        log_msg = {
            'epoch': epoch+1,
            'total_loss': avg_total,
            'recon_loss': avg_recon,
            'rq_loss': avg_rq,
            'time': epoch_time
        }
        print(log_msg)
        log_file.write(f"{log_msg}\n")
        log_file.flush()
        
        # TensorBoard
        writer.add_scalar('Loss/total', avg_total, epoch)
        writer.add_scalar('Loss/recon', avg_recon, epoch)
        writer.add_scalar('Loss/rq', avg_rq, epoch)
        
        # 保存最佳模型
        if avg_total < best_total_loss:
            best_total_loss = avg_total
            torch.save(model.state_dict(), save_dir / 'best_rqvae.pt')
            print(f"保存最佳模型至 {save_dir / 'best_rqvae.pt'}")
    
    # 5. 生成并保存全量item的语义ID
    print("开始生成语义ID...")
    model.load_state_dict(torch.load(save_dir / 'best_rqvae.pt'))
    model.eval()
    semantic_id_dict = {}  # {原始item ID: [c1, c2, ..., cN]}
    
    with torch.no_grad():
        full_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
        
        for batch in tqdm(full_dataloader, desc="生成语义ID"):
            tid_batch, emb_batch = batch
            emb_batch = emb_batch.to(args.device)
            
            # 生成语义ID
            semantic_ids = model.get_codebook(emb_batch)  # [B, num_codebooks]
            
            # 保存到字典（tid为原始item ID）
            for tid, sid in zip(tid_batch.numpy(), semantic_ids.cpu().numpy()):
                semantic_id_dict[str(tid)] = sid.tolist()  # 与Dataset中的indexer_i_rev对应
    
    # 保存语义ID
    semantic_id_path = save_dir / f'semantic_ids_{args.feature_id}.pkl'
    with open(semantic_id_path, 'wb') as f:
        pickle.dump(semantic_id_dict, f)
    print(f"语义ID保存完成：{semantic_id_path}，共{len(semantic_id_dict)}个item")
    
    writer.close()
    log_file.close()


if __name__ == '__main__':
    main()
