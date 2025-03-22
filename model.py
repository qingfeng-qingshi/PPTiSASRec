import pickle

import numpy as np
import torch
import sys
import torch.nn as nn
from torch import scatter

from GraphNetwork import GAT
from GraphNetwork import GGNN

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):
    """
    PointWiseFeedForward 类实现了一个带有残差连接的逐位置前馈网络。
    通过两次卷积和 ReLU 激活函数，模型能够在保持输入形状不变的情况下，对特征进行非线性变换，增强模型的表达能力。
    残差连接则有助于缓解梯度消失问题，提高模型的训练效果。
    """
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    """
    实现了一个时间感知的多头注意力机制，结合了时间矩阵和绝对位置编码，用于处理序列数据。
    通过多头注意力机制，模型能够更好地捕捉序列中的长期依赖关系和时间信息
    """
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs


class TiSASRec(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num, args):
        super(TiSASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.act = nn.ReLU()
        self.dropout = 0
        self.num_layers = 2
        self.alpha = 0.9
        self.args = args
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units, padding_idx=0)
        self.user_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.pop_emb = torch.nn.Embedding(9999, args.hidden_units, padding_idx=0)
        self.pop_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        #  生成了商品流行度、位置编码、时间矩阵的嵌入
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        # 初始化模块，便于后续的定义，类似list
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)  # 层归一化，在自注意力和前馈网络后使用
        # 添加GGNN层
        self.GGNN = GGNN(args.hidden_units, step=1)
        for _ in range(args.num_blocks):
            #归一化层
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            #加入上述模块构造网络
            self.attention_layernorms.append(new_attn_layernorm)
            #多头注意力加入模块，归一化
            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            #Relu和卷积提高网络的非线性处理方式，添加到模块中
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            # making GAT layers
            self.layers = torch.nn.ModuleList()
        #做了两个GAT，更新用户特征
        num_layers = 2
        for layer in range(num_layers):
            aggregator = GAT(args.hidden_units, args.hidden_units, act=self.act, dropout=self.dropout)
            self.layers.append(aggregator)
        #构建图神经网络，捕获社交关系
        self.build_graph()
    def build_graph(self):
        import pandas as pd
        friend_df = pd.read_csv('./data/social_relationships.csv',sep='\t',skiprows=1,names=['Follower', 'Followee', 'Weight'])
        edge_index = torch.tensor([friend_df['Follower'].values - 1, friend_df['Followee'].values - 1],
                                  dtype=torch.long)
        edge_weight = torch.tensor(friend_df['Weight'].values, dtype=torch.float)
        num_users = len(set(friend_df['Follower']).union(set(friend_df['Followee'])))
        #初始化一个随机特征矩阵，表示用户的特征向量
        x = torch.randn(num_users, self.args.hidden_units)  # 每个用户10维特征
        from torch_geometric.data import Data
        self.graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    def get_friends(self,user_ids):
        #使用padans读csv文件
        import pandas as pd
        friend_df = pd.read_csv('./data/social_relationships.csv',sep='\t',skiprows=1,names=['Follower', 'Followee', 'Weight'])
        friend_ids = []
        for user_id in user_ids:
            friend_ids.extend(friend_df[friend_df['Follower'] == user_id]['Followee'].values.tolist())
        #过滤friend_ids中大于self.user_num的元素
        friend_ids = [friend_id for friend_id in friend_ids if friend_id < self.user_num]

        return friend_ids

    # 将用户的行为序列转化为特征向量，输入到网络部分
    def seq2feats(self, user_ids, log_seqs, time_matrices):
              
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        #商品流行度与商品嵌入相加
        with open('商品流行度.pkl', 'rb') as file:
            item_popularity = pickle.load(file)
        # user_seqs = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        popularity_seqs = []
        for seq in log_seqs:
            popularity_seq = [min(int(item_popularity.get(item_id, 0) * 1000),9990) for item_id in seq]
            popularity_seqs.append(popularity_seq)
        pop_emb = self.pop_emb(torch.LongTensor(popularity_seqs).to(self.dev))
        pop_emb = self.pop_emb_dropout(pop_emb)
        seqs = seqs + pop_emb

        #每个位置生成一个位置编码，生成K V 
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        #时间间隔矩阵转换为嵌入层
        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        #创建掩码以忽略序列中的占位符 ；因果掩码（Causal Mask），确保在自注意力机制中不会看到未来的信息
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 更新的多头注意力层 （每次将序列作为query,进行注意力计算，query与输出相加，残差连接 ）
        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs) # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        #读取./data/social_relationships.csv文件
        user_friends = self.get_friends(user_ids)
        # 拿到userID及其朋友的userID后做embedding
        user_ids_tensor = torch.LongTensor(user_ids).to(self.dev)
        user_friends_tensor = torch.LongTensor(user_friends).to(self.dev)
        # 沿第二个维度（列）拼接
        combined_tensor = torch.cat((user_ids_tensor, user_friends_tensor), dim=0)
        combined_tensor = self.user_emb(combined_tensor)
        # 使用GAT更新用户特征
        for layer in self.layers:
            updated_user_features = layer([combined_tensor,combined_tensor])
        
        assert updated_user_features.shape[0] == len(user_ids) + len(user_friends), "Batch size mismatch"
        updated_user_features = updated_user_features[:len(user_ids)]
        # 将更新后的用户特征与序列特征结合，层归一化
        seqs = seqs + updated_user_features.unsqueeze(1).repeat(1, seqs.shape[1], 1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs): # for training
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_matrices, item_indices): # for inference
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
