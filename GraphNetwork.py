import torch
from torch import nn
import torch.nn.functional as F
class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=nn.ReLU(), **kwargs):
        super().__init__()

        self.act = act

        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.fc.weight = torch.nn.init.xavier_uniform_(self.fc.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        self_vecs, neigh_vecs = inputs

        if self.feat_drop is not None:
            self_vecs = self.feat_drop(self_vecs)
            neigh_vecs = self.feat_drop(neigh_vecs)

        # Reshape from [batch_size, depth] to [batch_size, 1, depth] for matmul.
        # self_vecs = torch.unsqueeze(self_vecs, 1)  # [batch, 1, embedding_size] 目标用户的节点表示
        neigh_self_vecs = torch.cat((neigh_vecs, self_vecs), dim=1)  # [batch, sample, embedding] 目标用户邻居的节点表示
        #第k个朋友对用户u的影响权重分
        score = self.softmax(torch.matmul(self_vecs, torch.transpose(neigh_vecs, 1, 0)))
        #用户Fu与其朋友兴趣的混合表示
        # alignment(score) shape is [batch_size, 1, depth]
        context = torch.squeeze(torch.matmul(score, neigh_vecs), dim=1)

        # [nodes] x [out_dim]
        output = self.act(self.fc(context))

        return output

class GGNN(nn.Module):
    # Gated Graph Neural Network
    def __init__(self, hidden_size, step=1):
        super(GGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size

        self.fc_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.fc_rzh_input = nn.Linear(2 * self.hidden_size, 3 * self.hidden_size, bias=True)
        self.fc_rz_old = nn.Linear(1 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.fc_h_old = nn.Linear(1 * self.hidden_size, 1 * self.hidden_size, bias=True)

    def aggregate(self, A, emb_items):
        h_input_in = self.fc_edge_in(torch.matmul(A[:, :, :A.shape[1]], emb_items))
        h_input_out = self.fc_edge_out(torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], emb_items))
        h_inputs = torch.cat([h_input_in, h_input_out], 2)

        r_input, z_input, h_input = self.fc_rzh_input(h_inputs).chunk(chunks=3, dim=2)
        r_old, z_old = self.fc_rz_old(emb_items).chunk(chunks=2, dim=2)

        reset = torch.sigmoid(r_old + r_input)
        update = torch.sigmoid(z_old + z_input)
        h = torch.tanh(h_input + self.fc_h_old(reset * emb_items))
        return (1 - update) * emb_items + update * h

    def forward(self, A, h_item):
        for i in range(self.step):
            h_item = self.aggregate(A, h_item)
        return h_item
