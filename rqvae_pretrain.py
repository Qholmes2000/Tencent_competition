import argparse
import os
import pickle
import time
import json  # 参考main.py使用json格式记录日志
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import MmEmbDataset, RQVAE


def get_args():
    parser = argparse.ArgumentParser()
    # 保持原有参数...
    parser.add_argument('--log_dir', default=os.environ.get('TRAIN_LOG_PATH', './logs'), type=str)
    parser.add_argument('--save_dir', default=os.environ.get('TRAIN_CKPT_PATH', './rqvae_pretrain'), type=str)
    parser.add_argument('--log_interval', default=100, type=int, help='每N步打印一次日志')  # 新增：日志间隔
    return parser.parse_args()


def main():
    args = get_args()
    # 日志和保存目录（强制使用平台环境变量）
    log_dir = Path(args.log_dir)
    save_dir = Path(args.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件（参考main.py的命名方式）
    log_file_path = log_dir / 'rqvae_pretrain.log'
    log_file = open(log_file_path, 'w', buffering=1)  # 行缓冲，实时写入
    print(f"RQVAE日志路径: {log_file_path}", flush=True)
    
    # 写入初始参数日志（参考main.py的json格式）
    param_log = {'args': vars(args), 'start_time': time.time()}
    log_file.write(json.dumps(param_log) + '\n')
    log_file.flush()
    
    # 1. 加载数据集（简化输出，无tqdm）
    print("开始加载数据集...", flush=True)
    try:
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
        data_log = {
            'step': 'dataset_loaded',
            'item_count': len(dataset),
            'batch_size': args.batch_size,
            'total_batches': len(dataloader)
        }
        log_file.write(json.dumps(data_log) + '\n')
        log_file.flush()
        print(f"数据集加载完成：{len(dataset)}个item，{len(dataloader)}个batch", flush=True)
    except Exception as e:
        error_log = {'step': 'dataset_error', 'error': str(e)}
        log_file.write(json.dumps(error_log) + '\n')
        log_file.flush()
        print(f"数据集加载失败: {e}", flush=True)
        return
    
    # 2. 初始化模型
    print("初始化RQVAE模型...", flush=True)
    try:
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
        model_log = {'step': 'model_initialized', 'model': str(model.__class__.__name__)}
        log_file.write(json.dumps(model_log) + '\n')
        log_file.flush()
        print("模型初始化成功", flush=True)
    except Exception as e:
        error_log = {'step': 'model_error', 'error': str(e)}
        log_file.write(json.dumps(error_log) + '\n')
        log_file.flush()
        print(f"模型初始化失败: {e}", flush=True)
        return
    
    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. 训练过程（无tqdm，用固定间隔打印）
    global_step = 0
    best_total_loss = float('inf')
    print(f"开始训练（共{args.num_epochs}轮）...", flush=True)
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_start = time.time()
        total_losses = 0.0
        recon_losses = 0.0
        rq_losses = 0.0
        
        # 遍历batch（无tqdm，手动计数）
        for batch_idx, batch in enumerate(dataloader):
            tid_batch, emb_batch = batch
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
            
            # 每log_interval步打印一次（参考main.py的步频）
            if (batch_idx + 1) % args.log_interval == 0:
                step_log = {
                    'step': 'train_step',
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'total_loss': total_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'rq_loss': rqvae_loss.item(),
                    'global_step': global_step
                }
                log_file.write(json.dumps(step_log) + '\n')
                log_file.flush()
                # 控制台打印简化信息
                print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss.item():.4f}", flush=True)
            
            global_step += 1
        
        # Epoch结束日志（参考main.py的epoch汇总）
        epoch_time = time.time() - epoch_start
        avg_total = total_losses / len(dataloader)
        avg_recon = recon_losses / len(dataloader)
        avg_rq = rq_losses / len(dataloader)
        
        epoch_log = {
            'step': 'train_epoch',
            'epoch': epoch,
            'avg_total_loss': avg_total,
            'avg_recon_loss': avg_recon,
            'avg_rq_loss': avg_rq,
            'time': epoch_time,
            'global_step': global_step
        }
        log_file.write(json.dumps(epoch_log) + '\n')
        log_file.flush()
        
        # 控制台打印epoch汇总
        print(f"\n===== Epoch {epoch} 完成 =====", flush=True)
        print(f"平均总损失: {avg_total:.4f}", flush=True)
        print(f"平均重构损失: {avg_recon:.4f}", flush=True)
        print(f"平均RQ损失: {avg_rq:.4f}", flush=True)
        print(f"耗时: {epoch_time:.2f}秒\n", flush=True)
        
        # 保存最佳模型
        if avg_total < best_total_loss:
            best_total_loss = avg_total
            model_path = save_dir / f"best_rqvae_epoch{epoch}.pt"
            torch.save(model.state_dict(), model_path)
            save_log = {
                'step': 'model_saved',
                'epoch': epoch,
                'best_loss': best_total_loss,
                'path': str(model_path)
            }
            log_file.write(json.dumps(save_log) + '\n')
            log_file.flush()
            print(f"保存最佳模型至: {model_path}", flush=True)
    
    # 5. 生成语义ID（同样无tqdm）
    print("开始生成语义ID...", flush=True)
    try:
        model.load_state_dict(torch.load(save_dir / f"best_rqvae_epoch{epoch}.pt", map_location=args.device))
        model.eval()
        semantic_id_dict = {}
        
        with torch.no_grad():
            # 遍历所有数据生成语义ID
            for batch_idx, batch in enumerate(dataloader):
                tid_batch, emb_batch = batch
                emb_batch = emb_batch.to(args.device)
                semantic_ids = model.get_codebook(emb_batch)  # 获取语义ID
                
                # 保存到字典
                for tid, sid in zip(tid_batch.numpy(), semantic_ids.cpu().numpy()):
                    semantic_id_dict[str(tid)] = sid.tolist()
                
                # 打印进度
                if (batch_idx + 1) % args.log_interval == 0:
                    print(f"生成语义ID: 已处理 {batch_idx+1}/{len(dataloader)} 个batch", flush=True)
        
        # 保存语义ID文件
        semantic_id_path = save_dir / f"semantic_ids_{args.feature_id}.pkl"
        with open(semantic_id_path, 'wb') as f:
            pickle.dump(semantic_id_dict, f)
        
        # 记录完成日志
        final_log = {
            'step': 'semantic_id_generated',
            'count': len(semantic_id_dict),
            'path': str(semantic_id_path),
            'end_time': time.time()
        }
        log_file.write(json.dumps(final_log) + '\n')
        log_file.flush()
        print(f"\n语义ID生成完成，共{len(semantic_id_dict)}个item，路径: {semantic_id_path}", flush=True)
    
    except Exception as e:
        error_log = {'step': 'semantic_id_error', 'error': str(e)}
        log_file.write(json.dumps(error_log) + '\n')
        log_file.flush()
        print(f"生成语义ID失败: {e}", flush=True)
        return
    
    # 结束
    log_file.close()
    print("===== RQVAE预训练全部完成 =====", flush=True)


if __name__ == '__main__':
    main()
