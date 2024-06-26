import dgl.function as dgl_fn
import torch.nn.functional as fn
import torch.nn as nn
import torch


class ESGNN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, re_eps=0.1, ir_eps=0.1, layer_num=2, iter_lowpass=1):
        super(ESGNN, self).__init__()
        self.re_eps = re_eps
        self.ir_eps = ir_eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.iter_lowpass = iter_lowpass

        # get norm
        deg = g.in_degrees().float()
        norm = torch.pow(deg.clamp(min=1), -0.5)
        g.ndata['d'] = norm.unsqueeze(1)
        self.g = g

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(ESGNN_Layer(self.g, hidden_dim, dropout))

        self.re_fc = nn.Linear(in_dim, hidden_dim // 2)
        self.ir_fc = nn.Linear(in_dim, hidden_dim // 2)
        self.cla = nn.Linear(hidden_dim // 2, out_dim)

    def _dropout(self, input):
        return fn.dropout(input, p=self.dropout, training=self.training)

    def forward(self, h):
        re_h = torch.relu(self.re_fc(h))
        ir_h = torch.relu(self.ir_fc(h))
        re_h, ir_h = self._dropout(re_h), self._dropout(ir_h)
        re_raw, ir_raw = re_h, ir_h
        for layer in self.layers:
            re_h, ir_h = layer(re_h, ir_h, self.iter_lowpass)
            re_h = self.re_eps * re_raw + (1 - self.re_eps) * re_h
            ir_h = self.ir_eps * ir_raw + (1 - self.ir_eps) * ir_h
        re_z, ir_z = re_h, ir_h
        re_logits = self.cla(re_h)
        ir_logits = self.cla(ir_h)
        return re_logits, ir_logits, re_z, ir_z

class ESGNN_Layer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(ESGNN_Layer, self).__init__()
        self.g = g
        self.dropout = dropout

        self.sub_gate = nn.Linear(2 * in_dim, 1)

    def _dropout(self, input):
        return fn.dropout(input, p=self.dropout, training=self.training)

    def edge_disentangling(self, edges):
        z = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        sub_scores = torch.tanh(self.sub_gate(z))
        sub_scores = self._dropout(sub_scores)
        re_s = (1 + sub_scores) / 2
        ir_s = (1 - sub_scores) / 2
        return {"re_s": re_s, "ir_s": ir_s}

    def norm_disentangling(self):
        self.g.update_all(dgl_fn.copy_e("re_s", "re_m"), dgl_fn.sum("re_m", "re_norm"))
        self.g.update_all(dgl_fn.copy_e("ir_s", "ir_m"), dgl_fn.sum("ir_m", "ir_norm"))
        self.g.ndata["re_norm"] = self.g.ndata["re_norm"].clamp(min=1).pow(-0.5)
        self.g.ndata["ir_norm"] = self.g.ndata["ir_norm"].clamp(min=1).pow(-0.5)

    def re_edge_applying(self, edges):
        return {"re_e": edges.data["re_s"] * edges.dst["re_norm"] * edges.src["re_norm"]}

    def ir_edge_applying(self, edges):
        return {"ir_e": edges.data["ir_s"] * edges.dst["ir_norm"] * edges.src["ir_norm"]}

    def forward(self, re_h, ir_h, iter_lowpass=1):
        # load data
        self.g.ndata.update({"re_h": re_h, "ir_h": ir_h, "h": torch.cat((re_h, ir_h), dim=1)})
        # disentangling
        self.g.apply_edges(self.edge_disentangling)
        self.norm_disentangling()
        # g-conv
        self.g.apply_edges(self.re_edge_applying)
        self.g.apply_edges(self.ir_edge_applying)
        for _ in range(iter_lowpass):
            self.g.update_all(dgl_fn.u_mul_e("re_h", "re_e", "re_"), dgl_fn.sum("re_", "re_h"))
            self.g.update_all(dgl_fn.u_mul_e("ir_h", "ir_e", "ir_"), dgl_fn.sum("ir_", "ir_h"))
        return self.g.ndata["re_h"], self.g.ndata["ir_h"]


class Label_Agree_Pred(nn.Module):
    def __init__(self, in_dim, dropout, metric_learnable=False):
        super(Label_Agree_Pred, self).__init__()
        if metric_learnable:
            self.pred_fc = nn.Linear(2 * in_dim, 1)
        self.dropout = dropout
        self.metric_learnable = metric_learnable

    def _dropout(self, input):
        return fn.dropout(input, p=self.dropout, training=self.training)

    def forward(self, g, n_key):
        with g.local_scope():
            g.ndata.update({"prob": g.ndata[n_key]})
            g.apply_edges(dgl_fn.u_dot_v("prob", "prob", "agree_e"))
            return g.edata["agree_e"]


class Ir_Consistency_Loss(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(Ir_Consistency_Loss, self).__init__()
        self.label_agree_predictor = Label_Agree_Pred(in_dim, dropout)
        self.g = g

    def laplacian_loss(self, n_key, e_key):
        self.g.apply_edges(dgl_fn.u_sub_v(n_key, n_key, "diff_e"))
        lap_loss = self.g.edata[e_key] * torch.pow(self.g.edata["diff_e"], 2).sum(dim=1, keepdim=True)
        return lap_loss.mean()

    def forward(self, re_, ir_h):
        self.g.ndata.update({"re_": re_, "ir_h": ir_h})
        agree_e = self.label_agree_predictor(self.g, "re_")
        self.g.edata.update({"dis_agree_e": 1 - agree_e})
        loss = self.laplacian_loss("ir_h", "dis_agree_e")
        return loss
