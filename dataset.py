import json
import pickle
import struct
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path


class MyDataset(torch.utils.data.Dataset):
    """用户序列数据集（支持语义ID特征）"""
    def __init__(self, data_dir, args, semantic_id_path=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        self.num_codebooks = args.num_codebooks if hasattr(args, 'num_codebooks') else 4  # 语义ID码本数量

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = self.load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}  # item reid -> 原始ID
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        # 加载语义ID字典（第一阶段输出）
        self.semantic_id_dict = {}
        if semantic_id_path:
            with open(semantic_id_path, 'rb') as f:
                self.semantic_id_dict = pickle.load(f)  # {原始item ID: [c1, c2, ..., cN]}

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        # 语义ID特征（新增）
        seq_sem_ids = np.zeros([self.maxlen + 1, self.num_codebooks], dtype=np.int32)
        pos_sem_ids = np.zeros([self.maxlen + 1, self.num_codebooks], dtype=np.int32)
        neg_sem_ids = np.zeros([self.maxlen + 1, self.num_codebooks], dtype=np.int32)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            
            # 填充特征（含语义ID）
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            
            # 序列信息
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            next_action_type[idx] = next_act_type if next_act_type is not None else 0
            seq_feat[idx] = feat
            
            # 语义ID（从字典获取）
            item_original_id = self.indexer_i_rev.get(i, str(i))
            seq_sem_ids[idx] = self.semantic_id_dict.get(item_original_id, [0]*self.num_codebooks)

            # 正样本
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                # 正样本语义ID
                pos_original_id = self.indexer_i_rev.get(next_i, str(next_i))
                pos_sem_ids[idx] = self.semantic_id_dict.get(pos_original_id, [0]*self.num_codebooks)
                
                # 负样本
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                # 负样本语义ID
                neg_original_id = self.indexer_i_rev.get(neg_id, str(neg_id))
                neg_sem_ids[idx] = self.semantic_id_dict.get(neg_original_id, [0]*self.num_codebooks)
            
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # 填充缺省值
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return (seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat, seq_sem_ids, pos_sem_ids, neg_sem_ids)

    def __len__(self):
        return len(self.seq_offsets)

    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        # 原有特征类型
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', 
            '120', '114', '112', '121', '115', '122', '116'
        ]
        # 新增语义ID特征（作为item_sparse）
        self.semantic_feat_ids = [f'semantic_{i}' for i in range(self.num_codebooks)]
        feat_types['item_sparse'].extend(self.semantic_feat_ids)
        
        # 其他特征类型
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        # 缺省值和统计信息
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            # 语义ID特征统计（使用码本大小）
            if feat_id.startswith('semantic_'):
                feat_statistics[feat_id] = 128  # 与codebook_size保持一致
            else:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array'] + feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual'] + feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        if feat is None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        # 补充缺失特征（含语义ID）
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]

        # 补充多模态emb
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if isinstance(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]], np.ndarray):
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        (seq, pos, neg, token_type, next_token_type, next_action_type, 
         seq_feat, pos_feat, neg_feat, seq_sem_ids, pos_sem_ids, neg_sem_ids) = zip(*batch)
        
        # 转换为张量
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_sem_ids = torch.from_numpy(np.array(seq_sem_ids))
        pos_sem_ids = torch.from_numpy(np.array(pos_sem_ids))
        neg_sem_ids = torch.from_numpy(np.array(neg_sem_ids))
        
        # 特征保持列表形式
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        
        return (seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat, seq_sem_ids, pos_sem_ids, neg_sem_ids)


class MyTestDataset(MyDataset):
    """测试数据集（继承修改）"""
    def __init__(self, data_dir, args, semantic_id_path=None):
        super().__init__(data_dir, args, semantic_id_path)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                value_list = []
                for v in feat_value:
                    value_list.append(0 if isinstance(v, str) else v)
                processed_feat[feat_id] = value_list
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        user_id = ""
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if isinstance(u, str):
                    user_id = u
                else:
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if isinstance(u, str):
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))
            if i and item_feat:
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        seq_sem_ids = np.zeros([self.maxlen + 1, self.num_codebooks], dtype=np.int32)  # 语义ID

        idx = self.maxlen
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            
            # 语义ID
            item_original_id = self.indexer_i_rev.get(i, str(i))
            seq_sem_ids[idx] = self.semantic_id_dict.get(item_original_id, [0]*self.num_codebooks)
            
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        return seq, token_type, seq_feat, user_id, seq_sem_ids

    def __len__(self):
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_id, seq_sem_ids = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_sem_ids = torch.from_numpy(np.array(seq_sem_ids))
        seq_feat = list(seq_feat)
        return seq, token_type, seq_feat, user_id, seq_sem_ids


def save_emb(emb, save_path):
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
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
