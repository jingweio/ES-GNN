from utils import to_undirected, remove_self_loops
from data_utils import eval_acc, eval_rocauc
from dataset import load_nc_dataset
from esgnn import ESGNN, Ir_Consistency_Loss
from parse import args

import torch.nn.functional as fn
import torch.optim as optim
import torch.nn as nn
import torch.autograd
import numpy as np
import torch
import dgl
import tempfile
import random
import time


# noinspection PyUnresolvedReferences
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EvalHelper:
    # noinspection PyUnresolvedReferences
    def __init__(self, args):
        use_cuda = torch.cuda.is_available() and not args.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        # load dataset
        dataset = load_nc_dataset(args.dataset, args.sub_dataset)
        # load splits
        if args.dataset in ['chameleon', 'squirrel', 'film', 'twitch-e', 'cora', 'citeseer', 'pubmed', 'polblogs']:
            if args.dataset == "twitch-e":
                assert args.sub_dataset in ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']
            split_dic = np.load(f"{args.DATAPATH}split/LR{args.label_rate}/LR{args.label_rate}_{args.dataset}_{args.sub_dataset}_splits.npy",
                                allow_pickle=True)[args.split_index]
        elif args.dataset == "etg_syn_hom":
            split_dic = dataset.graph[f"LR{args.label_rate}_splits"][args.split_index]
        else:
            raise ValueError('Invalid method')
        trn_idx, val_idx, tst_idx = np.array(split_dic["trn_idx"]), np.array(split_dic["val_idx"]), np.array(split_dic["tst_idx"])
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0
        # pre-processing
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        # n = dataset.graph['num_nodes']
        c = dataset.label.max().item() + 1
        d = dataset.graph['node_feat'].shape[1]
        dataset.graph["edge_index"] = remove_self_loops(dataset.graph["edge_index"])[0]
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        # to-cuda
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        dataset.label = dataset.label.to(dev)
        dataset.graph['edge_index'] = dataset.graph['edge_index'].to(dev)
        dataset.graph['node_feat'] = dataset.graph['node_feat'].to(dev)

        edge = dataset.graph["edge_index"]
        g = dgl.graph((edge[0], edge[1]))
        model = ESGNN(g, d, args.hidden_channels, c, args.dropout, re_eps=args.re_eps, ir_eps=args.ir_eps, layer_num=args.num_layers).to(dev)

        all_params = model.parameters()
        self.ir_con_loss_fn = Ir_Consistency_Loss(g, args.hidden_channels // 2, args.dropout).to(dev)
        all_params = list(all_params) + list(self.ir_con_loss_fn.parameters())
        optmz = optim.Adam(all_params, lr=args.lr, weight_decay=args.reg)

        # considering the class-imbalance problem in Twitch-DE
        if args.rocauc or args.dataset == 'twitch-e':
            loss_fn = nn.BCEWithLogitsLoss()
            eval_fn = eval_rocauc
        else:
            loss_fn = nn.NLLLoss()
            eval_fn = eval_acc

        self.dataset = dataset
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz = model, optmz
        self.loss_fn, self.eval_fn = loss_fn, eval_fn
        self.args = args

    def before_loss(self, args, out):
        # considering the class-imbalance problem in Twitch-DE
        if args.rocauc or args.dataset == 'twitch-e':
            true_label = fn.one_hot(self.dataset.label, self.dataset.label.max() + 1).type(out.dtype)
        else:
            true_label = self.dataset.label
            out = fn.log_softmax(out, dim=1)
        return out, true_label

    def to_onehot(self, input):
        oh_input = torch.zeros(input.shape[0], input.max() + 1).to(input.device)
        val_idxes = torch.where(input >= 0)[0]
        oh_input[val_idxes] = fn.one_hot(input[val_idxes].long(), input[val_idxes].long().max() + 1).to(oh_input.dtype)
        return oh_input

    def run_epoch(self, args):
        self.model.train()
        self.optmz.zero_grad()
        re_logits, ir_logits, re_feat, ir_feat = self.model(self.dataset.graph["node_feat"])
        # prediction loss
        re_out, true_label = self.before_loss(args, re_logits)
        pred_loss = self.loss_fn(re_out[self.trn_idx], true_label.squeeze(1)[self.trn_idx])
        # Irrelevant Consistency Regularization
        valtst_idx = torch.cat((self.val_idx, self.tst_idx), dim=0)
        masked_pred = torch.zeros_like(re_logits)
        masked_pred[self.trn_idx] = self.to_onehot(self.dataset.label.squeeze(1)).float()[self.trn_idx]
        masked_pred[valtst_idx] = fn.softmax(re_logits[valtst_idx], dim=1)
        ir_con_loss = self.ir_con_loss_fn(masked_pred, ir_feat)
        loss = pred_loss + args.ir_con_lambda * ir_con_loss
        loss.backward()
        self.optmz.step()
        print("epoch-loss: {:.4f}, pred-loss: {:.4f}, ir-con-loss: {:.4f}".format(loss.item(), pred_loss.item(), ir_con_loss.item()))
        return loss.item()

    def evaluate(self):
        self.model.eval()
        out, _, _, _ = self.model(self.dataset.graph["node_feat"])
        trn_acc = self.eval_fn(self.dataset.label[self.trn_idx], out[self.trn_idx])
        val_acc = self.eval_fn(self.dataset.label[self.val_idx], out[self.val_idx])
        tst_acc = self.eval_fn(self.dataset.label[self.tst_idx], out[self.tst_idx])
        return trn_acc, val_acc, tst_acc


# noinspection PyUnresolvedReferences
def train_and_eval(args):
    # fix random initialization
    set_rng_seed(args.rnd_seed)
    # build model
    agent = EvalHelper(args)
    # trn and val
    wait_cnt, best_epoch = 0, 0
    best_val_acc = 0.0
    best_model_sav = tempfile.TemporaryFile()
    ct_ls = []
    for t in range(args.nepoch):
        cur_time = time.time()
        agent.run_epoch(args)
        ct_ls.append(time.time() - cur_time)
        trn_acc, val_acc, tst_acc = agent.evaluate()
        print("epoch: {}/{}, trn-acc={:.4f}%, val-acc={:.4f}%, tst-acc={:.4f}%".format(
            t + 1, args.nepoch, trn_acc * 100, val_acc * 100, tst_acc * 100))
        # training with early-stop
        if val_acc > best_val_acc:
            wait_cnt = 0
            best_val_acc = val_acc
            best_model_sav.close()
            best_model_sav = tempfile.TemporaryFile()
            torch.save(agent.model.state_dict(), best_model_sav)
            best_epoch = t + 1
        else:
            wait_cnt += 1
            if wait_cnt > args.early:
                break
    # final results
    print("Load selected model ...")
    best_model_sav.seek(0)
    agent.model.load_state_dict(torch.load(best_model_sav))
    trn_acc, val_acc, tst_acc = agent.evaluate()
    print("trn-acc={:.4f}%, val-acc={:.4f}%, tst-acc={:.4f}%, avg-epoch-time={:.4f}".format(trn_acc * 100, val_acc * 100, tst_acc * 100, np.mean(ct_ls)))
    return val_acc, tst_acc, best_epoch


def run(args):
    val_acc, tst_acc, selected_epoch = train_and_eval(args)
    print("val_acc={:.4f}%, tst_acc={:.4f}%, selected_epoch={}".format(val_acc * 100, tst_acc * 100, selected_epoch))


def main():
    run(args)


if __name__ == '__main__':
    main()
