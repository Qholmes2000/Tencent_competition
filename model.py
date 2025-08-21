import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from pathlib import Path


class MmEmbDataset(torch.utils.data.Dataset):
    """多模态emb数据集（用于RQVAE预训练）"""
    def __init__(self, data_dir, feature_id):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mm_emb_id = [feature_id]
        self.mm_emb_dict = self._load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_id)

        self.mm_emb = self.mm_emb_dict[self.mm_emb_id[0]]
        self.tid_list, self.emb_list = list(self.mm_emb.keys()), list(self.mm_emb.values())
        self.emb_list = [torch.tensor(emb, dtype=torch.float32) for emb in self.emb_list]

        assert len(self.tid_list) == len(self.emb_list)
        self.item_cnt = len(self.tid_list)

    def _load_mm_emb(self, mm_path, feat_ids):
        """加载多模态emb（复用Dataset.py中的逻辑）"""
        SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        mm_emb_dict = {}
        for feat_id in feat_ids:
            shape = SHAPE_DICT[feat_id]
            emb_dict = {}
            if feat_id != '81':
                try:
                    base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                    for json_file in base_path.glob('*.json'):
                        with open(json_file, 'r', encoding='utf-8') as file:
                            for line in file:
                                data_dict_origin = json.loads(line.strip())
                                insert_emb = data_dict_origin['emb']
                                if isinstance(insert_emb, list):
                                    insert_emb = np.array(insert_emb, dtype=np.float32)
                                data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                                emb_dict.update(data_dict)
                except Exception as e:
                    print(f"transfer error: {e}")
            if feat_id == '81':
                with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                    emb_dict = pickle.load(f)
            mm_emb_dict[feat_id] = emb_dict
            print(f'Loaded #{feat_id} mm_emb')
        return mm_emb_dict

    def __getitem__(self, index):
        tid = torch.tensor(int(self.tid_list[index]), dtype=torch.long)
        emb = self.emb_list[index]
        return tid, emb

    def __len__(self):
        return self.item_cnt

    @staticmethod
    def collate_fn(batch):
        tid, emb = zip(*batch)
        tid_batch, emb_batch = torch.stack(tid, dim=0), torch.stack(emb, dim=0)
        return tid_batch, emb_batch


## Kmeans聚类
def kmeans(data, n_clusters, kmeans_iters):
    km = KMeans(n_clusters=n_clusters, max_iter=kmeans_iters, n_init="auto")
    data_cpu = data.detach().cpu()
    np_data = data_cpu.numpy()
    km.fit(np_data)
    return torch.tensor(km.cluster_centers_), torch.tensor(km.labels_)


## 平衡Kmeans
class BalancedKmeans(torch.nn.Module):
    def __init__(self, num_clusters: int, kmeans_iters: int, tolerance: float, device: str):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.tolerance = tolerance
        self.device = device
        self._codebook = None

    def _compute_distances(self, data):
        return torch.cdist(data, self._codebook)

    def _assign_clusters(self, dist):
        samples_cnt = dist.shape[0]
        samples_labels = torch.zeros(samples_cnt, dtype=torch.long, device=self.device)
        clusters_cnt = torch.zeros(self.num_clusters, dtype=torch.long, device=self.device)

        sorted_indices = torch.argsort(dist, dim=-1)
        for i in range(samples_cnt):
            for j in range(self.num_clusters):
                cluster_idx = sorted_indices[i, j]
                if clusters_cnt[cluster_idx] < samples_cnt // self.num_clusters:
                    samples_labels[i] = cluster_idx
                    clusters_cnt[cluster_idx] += 1
                    break
        return samples_labels

    def _update_codebook(self, data, samples_labels):
        _new_codebook = []
        for i in range(self.num_clusters):
            cluster_data = data[samples_labels == i]
            if len(cluster_data) > 0:
                _new_codebook.append(cluster_data.mean(dim=0))
            else:
                _new_codebook.append(self._codebook[i])
        return torch.stack(_new_codebook)

    def fit(self, data):
        num_emb, codebook_emb_dim = data.shape
        data = data.to(self.device)
        indices = torch.randperm(num_emb)[: self.num_clusters]
        self._codebook = data[indices].clone()

        for _ in range(self.kmeans_iters):
            dist = self._compute_distances(data)
            samples_labels = self._assign_clusters(dist)
            _new_codebook = self._update_codebook(data, samples_labels)
            if torch.norm(_new_codebook - self._codebook) < self.tolerance:
                break
            self._codebook = _new_codebook
        return self._codebook, samples_labels

    def predict(self, data):
        data = data.to(self.device)
        dist = self._compute_distances(data)
        samples_labels = self._assign_clusters(dist)
        return samples_labels


## RQVAE编码器
class RQEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: list, latent_dim: int):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        in_dim = input_dim
        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim
        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, latent_dim), torch.nn.ReLU()))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


## RQVAE解码器
class RQDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: list, output_dim: int):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        in_dim = latent_dim
        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim
        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, output_dim), torch.nn.ReLU()))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


## 向量量化嵌入（生成语义ID）
class VQEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_clusters,
        codebook_emb_dim: int,
        kmeans_method: str,
        kmeans_iters: int,
        distances_method: str,
        device: str,
    ):
        super(VQEmbedding, self).__init__(num_clusters, codebook_emb_dim)
        self.num_clusters = num_clusters
        self.codebook_emb_dim = codebook_emb_dim
        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.device = device

    def _create_codebook(self, data):
        if self.kmeans_method == 'kmeans':
            _codebook, _ = kmeans(data, self.num_clusters, self.kmeans_iters)
        elif self.kmeans_method == 'bkmeans':
            BKmeans = BalancedKmeans(
                num_clusters=self.num_clusters, kmeans_iters=self.kmeans_iters, tolerance=1e-4, device=self.device
            )
            _codebook, _ = BKmeans.fit(data)
        else:
            _codebook = torch.randn(self.num_clusters, self.codebook_emb_dim)
        _codebook = _codebook.to(self.device)
        assert _codebook.shape == (self.num_clusters, self.codebook_emb_dim)
        self.codebook = torch.nn.Parameter(_codebook)

    @torch.no_grad()
    def _compute_distances(self, data):
        _codebook_t = self.codebook.t()
        assert _codebook_t.shape == (self.codebook_emb_dim, self.num_clusters)
        assert data.shape[-1] == self.codebook_emb_dim

        if self.distances_method == 'cosine':
            data_norm = F.normalize(data, p=2, dim=-1)
            _codebook_t_norm = F.normalize(_codebook_t, p=2, dim=0)
            distances = 1 - torch.mm(data_norm, _codebook_t_norm)
        else:
            data_norm_sq = data.pow(2).sum(dim=-1, keepdim=True)
            _codebook_t_norm_sq = _codebook_t.pow(2).sum(dim=0, keepdim=True)
            distances = torch.addmm(data_norm_sq + _codebook_t_norm_sq, data, _codebook_t, beta=1.0, alpha=-2.0)
        return distances

    @torch.no_grad()
    def _create_semantic_id(self, data):
        distances = self._compute_distances(data)
        _semantic_id = torch.argmin(distances, dim=-1)
        return _semantic_id

    def _update_emb(self, _semantic_id):
        update_emb = super().forward(_semantic_id)
        return update_emb

    def forward(self, data):
        self._create_codebook(data)
        _semantic_id = self._create_semantic_id(data)
        update_emb = self._update_emb(_semantic_id)
        return update_emb, _semantic_id


## 残差量化器
class RQ(torch.nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: list,
        codebook_emb_dim,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        assert len(self.codebook_size) == self.num_codebooks
        self.codebook_emb_dim = codebook_emb_dim
        self.shared_codebook = shared_codebook

        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.loss_beta = loss_beta
        self.device = device

        if self.shared_codebook:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[0],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for _ in range(self.num_codebooks)
                ]
            )
        else:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[idx],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for idx in range(self.num_codebooks)
                ]
            )

    def quantize(self, data):
        res_emb = data.detach().clone()
        vq_emb_list, res_emb_list = [], []
        semantic_id_list = []
        vq_emb_aggre = torch.zeros_like(data)

        for i in range(self.num_codebooks):
            vq_emb, _semantic_id = self.vqmodules[i](res_emb)
            res_emb -= vq_emb
            vq_emb_aggre += vq_emb
            res_emb_list.append(res_emb)
            vq_emb_list.append(vq_emb_aggre)
            semantic_id_list.append(_semantic_id.unsqueeze(dim=-1))

        semantic_id_list = torch.cat(semantic_id_list, dim=-1)
        return vq_emb_list, res_emb_list, semantic_id_list

    def _rqvae_loss(self, vq_emb_list, res_emb_list):
        rqvae_loss_list = []
        for idx, quant in enumerate(vq_emb_list):
            loss1 = (res_emb_list[idx].detach() - quant).pow(2.0).mean()
            loss2 = (res_emb_list[idx] - quant.detach()).pow(2.0).mean()
            partial_loss = loss1 + self.loss_beta * loss2
            rqvae_loss_list.append(partial_loss)
        return torch.sum(torch.stack(rqvae_loss_list))

    def forward(self, data):
        vq_emb_list, res_emb_list, semantic_id_list = self.quantize(data)
        rqvae_loss = self._rqvae_loss(vq_emb_list, res_emb_list)
        return vq_emb_list, semantic_id_list, rqvae_loss


## RQVAE完整模型（支持生成语义ID）
class RQVAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: list,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: list,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        super().__init__()
        self.encoder = RQEncoder(input_dim, hidden_channels, latent_dim).to(device)
        self.decoder = RQDecoder(latent_dim, hidden_channels[::-1], input_dim).to(device)
        self.rq = RQ(
            num_codebooks,
            codebook_size,
            latent_dim,
            shared_codebook,
            kmeans_method,
            kmeans_iters,
            distances_method,
            loss_beta,
            device,
        ).to(device)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z_vq):
        if isinstance(z_vq, list):
            z_vq = z_vq[-1]
        return self.decoder(z_vq)

    def compute_loss(self, x_hat, x_gt, rqvae_loss):
        recon_loss = F.mse_loss(x_hat, x_gt, reduction="mean")
        total_loss = recon_loss + rqvae_loss
        return recon_loss, rqvae_loss, total_loss

    def get_codebook(self, x_gt):  # 公开方法：生成语义ID
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        return semantic_id_list  # [batch_size, num_codebooks]

    def forward(self, x_gt):
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss


## Baseline模型（支持语义ID稀疏特征）
class BaselineModel(nn.Module):
    def __init__(self, usernum, itemnum, feat_statistics, feat_types, args):
        super().__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.hidden_units = args.hidden_units
        self.num_heads = args.num_heads
        self.dropout_rate = args.dropout_rate
        self.norm_first = args.norm_first
        self.device = args.device

        # 特征类型定义
        self.feat_types = feat_types
        self.semantic_feat_ids = [f'semantic_{i}' for i in range(args.num_codebooks)]  # 语义ID特征ID

        # 1. 嵌入层定义
        # 用户/物品基础嵌入
        self.user_emb = nn.Embedding(usernum + 1, self.hidden_units, padding_idx=0)
        self.item_emb = nn.Embedding(itemnum + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, self.hidden_units, padding_idx=0)  # 位置嵌入

        # 稀疏特征嵌入（含语义ID）
        self.sparse_emb = nn.ModuleDict()
        # 原有稀疏特征
        for feat_id in self.feat_types['user_sparse'] + self.feat_types['item_sparse']:
            self.sparse_emb[feat_id] = nn.Embedding(feat_statistics[feat_id] + 1, self.hidden_units // 4, padding_idx=0)
        # 语义ID稀疏特征（每个码本单独嵌入）
        for feat_id in self.semantic_feat_ids:
            self.sparse_emb[feat_id] = nn.Embedding(args.codebook_size[0] + 1, self.hidden_units // 4, padding_idx=0)

        # 2. Transformer层
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_units,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_units * 4,
            dropout=self.dropout_rate,
            norm_first=self.norm_first,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=args.num_blocks)

        # 3. 输出层
        self.output_layer = nn.Linear(self.hidden_units, 1)

    def forward(self, seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat, seq_sem_ids=None, pos_sem_ids=None, neg_sem_ids=None):
        # 序列长度
        seq_len = seq.size(1)
        batch_size = seq.size(0)

        # 1. 基础嵌入（序列）
        seq_emb = self.item_emb(seq)  # [B, L, H]
        pos_emb = self.pos_emb(torch.arange(1, seq_len + 1, device=self.device).unsqueeze(0).repeat(batch_size, 1))  # [B, L, H]
        seq_emb = seq_emb + pos_emb  # 叠加位置嵌入

        # 2. 稀疏特征嵌入（含语义ID）
        # 序列特征嵌入
        for i in range(seq_len):
            # 原有稀疏特征
            for feat_id in self.feat_types['item_sparse']:
                feat_val = torch.tensor([seq_feat[b][i].get(feat_id, 0) for b in range(batch_size)], device=self.device)
                seq_emb[:, i] += self.sparse_emb[feat_id](feat_val)
            # 语义ID特征
            if seq_sem_ids is not None:
                for idx, feat_id in enumerate(self.semantic_feat_ids):
                    sem_val = seq_sem_ids[:, i, idx]  # [B]
                    seq_emb[:, i] += self.sparse_emb[feat_id](sem_val)

        # 3. Transformer编码
        mask = (seq == 0).unsqueeze(1).repeat(1, seq_len, 1)  # [B, L, L]
        seq_encoded = self.transformer(seq_emb, src_mask=mask)  # [B, L, H]

        # 4. 预测逻辑（取最后一个有效位置）
        last_idx = (seq != 0).sum(dim=1) - 1  # [B]
        last_encoded = seq_encoded[torch.arange(batch_size), last_idx]  # [B, H]

        # 正样本/负样本嵌入
        pos_emb = self.item_emb(pos)[:, -1]  # [B, H]
        neg_emb = self.item_emb(neg)[:, -1]  # [B, H]

        # 正/负样本稀疏特征
        for feat_id in self.feat_types['item_sparse']:
            pos_val = torch.tensor([pos_feat[b][-1].get(feat_id, 0) for b in range(batch_size)], device=self.device)
            neg_val = torch.tensor([neg_feat[b][-1].get(feat_id, 0) for b in range(batch_size)], device=self.device)
            pos_emb += self.sparse_emb[feat_id](pos_val)
            neg_emb += self.sparse_emb[feat_id](neg_val)
        # 语义ID特征
        if pos_sem_ids is not None and neg_sem_ids is not None:
            for idx, feat_id in enumerate(self.semantic_feat_ids):
                pos_sem_val = pos_sem_ids[:, -1, idx]  # [B]
                neg_sem_val = neg_sem_ids[:, -1, idx]  # [B]
                pos_emb += self.sparse_emb[feat_id](pos_sem_val)
                neg_emb += self.sparse_emb[feat_id](neg_sem_val)

        # 计算预测分数（点积）
        pos_logits = (last_encoded * pos_emb).sum(dim=-1)  # [B]
        neg_logits = (last_encoded * neg_emb).sum(dim=-1)  # [B]

        return pos_logits.unsqueeze(1), neg_logits.unsqueeze(1), None  # 第三项为兼容旧代码
