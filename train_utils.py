import torch
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
import warnings
import pytorch_warmup as warmup
from torch_geometric.utils import index_to_mask
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from data import read_and_unpkl
from utils import norm_entropy
import numpy as np
import editdistance
import ast



class WarmupExpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, gamma=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.gamma = gamma
        super(WarmupExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]

    



def get_optimizer(args, model):
    if args.model_name == 'LP':
        return None, None
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        scheduler = None 
    elif args.optim == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupExpLR(optimizer, args.warmup, total_epochs=args.epochs, gamma=args.lr_gamma)
    return optimizer, scheduler


def train(model, data, optimizer, loss_fn, train_mask, val_mask):
    model.train()
    optimizer.zero_grad()
    preds = model(data)
    if len(data.y.shape) != 1:
        y = data.y.squeeze(1)
    else:
        y = data.y
    train_loss = loss_fn(preds[train_mask], y[train_mask])
    train_loss.backward()
    optimizer.step()
    val_loss = loss_fn(preds[val_mask], y[val_mask])
    val_acc = test(model, data, False, val_mask)
    return train_loss, val_loss, val_acc


def batch_train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch_size, n_id, edge_index = batch.batch_size, batch.n_id, batch.edge_index
        # data = data.to(device)
        optimizer.zero_grad()
        batch.edge_index = batch.edge_index.to(device)
        out = model(batch)[:batch_size]
        y = batch.y[:batch_size].squeeze()
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def to_inductive(data, msk_index = 0):
    data = data.clone()
    mask = data.train_masks[msk_index]
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = mask[mask]
    data.test_masks = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data



@torch.no_grad()
def batch_test(model, data, evaluator, subgraph_loader, device, mask):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_pred = out.argmax(dim=-1, keepdim=True)

    # import ipdb; ipdb.set_trace()
    if len(data.y.shape) == 1:
        y_true = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y_true = data.y

    test_acc = evaluator.eval({
        'y_true': y_true[mask],
        'y_pred': y_pred[mask]
    })['acc']

    return test_acc



@torch.no_grad()
def topk_test(model, data, mask, topk = 3, need_batch = False, subgraph_loader = None):
    model.eval()
    # model.model.initialized = False
    if not need_batch:
        out = model(data)
        y_pred = out.argmax(dim=-1, keepdim=True)
    else:
        out = model.inference(data.x, subgraph_loader, device)
        y_true = data.y
        y_pred = out.argmax(dim=-1, keepdim=True)
    r_y_pred = y_pred.reshape(-1)
    confidence = out.gather(1, r_y_pred.unsqueeze(1)).reshape(-1)
    data.confidence = confidence
    sorted_conf_idx = torch.argsort(data.confidence)
    full_length = data.x.shape[0]
    com_res = data.y.view(-1, 1).expand_as(out.topk(3,1).values).eq(out.topk(3,1).indices).sum(-1).to(torch.bool)
    low_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[:full_length // 3], size=full_length)
    med_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length // 3 : full_length * 2 // 3], size=full_length)
    high_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length * 2 // 3:], size=full_length)

    y_1 = y_pred.reshape(-1)
    true_mask = (y_1 == data.y)
    false_mask = ~true_mask

    evaluator = Evaluator(name='ogbn-arxiv')
    top3_low_acc = torch.sum(com_res[mask & low_confidence_sorted_conf_mask]) / com_res[mask & low_confidence_sorted_conf_mask].shape[0]
    top3_med_acc = torch.sum(com_res[mask & med_confidence_sorted_conf_mask]) / com_res[mask & med_confidence_sorted_conf_mask].shape[0]
    top3_high_acc = torch.sum(com_res[mask & high_confidence_sorted_conf_mask]) / com_res[mask & high_confidence_sorted_conf_mask].shape[0]
    # true_acc = torch.sum(com_res[mask & true_mask]) / com_res[mask & true_mask].shape[0]
    
    res = data.y.view(-1).eq(r_y_pred)
    top1_low_acc = torch.sum(res[mask & low_confidence_sorted_conf_mask]) / res[mask & low_confidence_sorted_conf_mask].shape[0]
    top1_med_acc = torch.sum(res[mask & med_confidence_sorted_conf_mask]) / res[mask & med_confidence_sorted_conf_mask].shape[0]
    top1_high_acc = torch.sum(res[mask & high_confidence_sorted_conf_mask]) / res[mask & high_confidence_sorted_conf_mask].shape[0]
    # top1_low_acc = torch.sum()
    top3_false_acc = torch.sum(com_res[mask & false_mask]) / com_res[mask & false_mask].shape[0]
    total_acc = torch.sum(com_res[mask]) / com_res[mask].shape[0]
    print("Top3 Accuracy on low confidence nodes: {}\n".format(top3_low_acc.item()))
    print("Top3 Accuracy on medium confidence nodes: {}\n".format(top3_med_acc.item()))
    print("Top3 Accuracy on high confidence nodes: {}\n".format(top3_high_acc.item()))
    print("Top1 Accuracy on low confidence nodes: {}\n".format(top1_low_acc.item()))
    print("Top1 Accuracy on medium confidence nodes: {}\n".format(top1_med_acc.item()))
    print("Top1 Accuracy on high confidence nodes: {}\n".format(top1_high_acc.item()))
    print("Top3 Accuracy on gnn false nodes: {}\n".format(top3_false_acc.item()))
    return top3_low_acc.item(), top3_med_acc.item(), top3_high_acc.item(), total_acc.item()



@torch.no_grad()
def confidence_test(model, data, mask):
    model.eval()
    # model.model.initialized = False
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    r_y_pred = y_pred.reshape(-1)
    confidence = out.gather(1, r_y_pred.unsqueeze(1)).reshape(-1)
    data.confidence = confidence
    sorted_conf_idx = torch.argsort(data.confidence)
    full_length = data.x.shape[0]
    low_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[:full_length // 3], size=full_length)
    med_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length // 3 : full_length * 2 // 3], size=full_length)
    high_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length * 2 // 3:], size=full_length)
    # ground_truth = data.y.cpu()
    # true_mask = data.y.cpu() == y_pred.cpu()
    # false_mask = data.y.cpu() != y_pred.cpu()

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y
    
    y_1 = y_pred.reshape(-1)
    true_mask = (y_1 == data.y)
    false_mask = ~true_mask

    evaluator = Evaluator(name='ogbn-arxiv')
    low_acc = evaluator.eval({
        'y_true': y[mask | low_confidence_sorted_conf_mask],
        'y_pred': y_pred[mask | low_confidence_sorted_conf_mask],
    })['acc']
    
    med_acc = evaluator.eval({
        'y_true': y[mask | med_confidence_sorted_conf_mask],
        'y_pred': y_pred[mask | med_confidence_sorted_conf_mask],
    })['acc']

    high_acc = evaluator.eval({
        'y_true': y[mask | high_confidence_sorted_conf_mask],
        'y_pred': y_pred[mask | high_confidence_sorted_conf_mask],
    })['acc']


    true_acc = evaluator.eval({
        'y_true': y[mask | true_mask],
        'y_pred': y_pred[mask | true_mask],
    })['acc']


    false_acc = evaluator.eval({
        'y_true': y[mask | false_mask],
        'y_pred': y_pred[mask | false_mask],
    })['acc']

    print(true_acc, false_acc)

    return low_acc, med_acc, high_acc

@torch.no_grad()
def test(model, data, return_embeds, mask):
    model.eval()
    # model.model.initialized = False
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    acc = evaluator.eval({
        'y_true': y[mask],
        'y_pred': y_pred[mask],
    })['acc']

    
    if not return_embeds:
        return acc, None
    else:
        return acc, out


def loss_kd(all_out, teacher_all_out, outputs, labels, teacher_outputs,
            alpha, temperature):
    """
    loss function for Knowledge Distillation (KD)
    """

    T = temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss

def loss_kd_only(all_out, teacher_all_out, temperature):
    T = temperature

    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)

    return D_KL



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True




# def glem(num_nodes, hidden_num, n_labels,  dataset = 'cora', gnn_model_name = 'GCN', feature_name = 'sbert', lm_output_path = './lmoutput', gnn_output_path = './output', setting = 'fixed'):
#     ## pretrain lm and get the embeddings
#     lm_output_embedding_path = osp.join(lm_output_path, f"{dataset}_finetune_{setting}.emb")
#     lm_output_pred_path = osp.join(lm_output_path, f"{dataset}_finetune_{setting}.pred")
#     lm_emb_np = np.memmap(lm_output_embedding_path, dtype=np.float16, mode='r',
#                              shape=(num_nodes, hidden_dim))
#     lm_pred = np.memmap(lm_output_pred_path, dtype=np.float16, mode='r',
#                               shape=(num_nodes, n_labels))
#     lm_emb = torch.tensor(emb, dtype=torch.float32)
#     lm_pred = torch.tensor(pred, dtype=torch.float32)
#     ## pretrain gnn and get the embeddings
#     gnn_pred_path = osp.join(gnn_output_path, f"{gnn_model_name}_{dataset}_{feature_name}.pkl")
#     gnn_pred = read_and_unpkl(gnn_pred_path)


def tensor_intersection(tensor1, tensor2):
    set1 = set(tensor1.numpy().flatten())
    set2 = set(tensor2.numpy().flatten())
    
    intersection = set1 & set2

    return torch.tensor(list(intersection))

@torch.no_grad()
def llm_pseudo_label(data, logits, budget = 100, train_val_ratio = 3, strategy = 1):
    """
        train_val_ratio:  new train : new val
        Strategy 1: totally random
        Strategy 2: each class random
        Strategy 3: confidence based
        Strategy 4: class confidence based
        Strategy 5: use prompt to test llm's confidence
    """
    ## data is the low labeling rate data
    node_idx = torch.arange(data.x.shape[0])
    test_mask = data.test_masks[0].cpu()
    data = data.cpu()
    test_idx = node_idx[test_mask]
    if strategy == 1:
        selected_test_idx = torch.randperm(test_idx.shape[0])[:budget]
    elif strategy == 2:
        num_of_class = data.y.max().item() + 1
        per_class = budget // num_of_class
        selected_test_idx = []
        count = [0 for _ in range(num_of_class)]
        rand_node_idx = torch.randperm(test_idx.shape[0])
        for i in rand_node_idx:
            if i not in test_idx: continue
            lbl = data.y[i].item()
            if count[lbl] < per_class:
                selected_test_idx.append(i.item())
                count[lbl] += 1
            if min(count) == per_class: break
        selected_test_idx = torch.LongTensor(selected_test_idx)
    elif strategy == 3:
        norm_entro = norm_entropy(logits)
        test_idx_set = set(test_idx.tolist())
        sorted_idx = torch.argsort(norm_entro).tolist()
        intersection = [i for i in sorted_idx if i in test_idx_set]
        selected_test_idx = torch.LongTensor(intersection[:budget])
    elif strategy == 4:
        num_of_class = data.y.max().item() + 1
        per_class = budget // num_of_class
        count = [0 for _ in range(num_of_class)]
        norm_entro = norm_entropy(logits)
        test_idx_set = set(test_idx.tolist())
        sorted_idx = torch.argsort(norm_entro).tolist()
        for i in sorted_idx:
            if i not in test_idx_set: continue
            lbl = data.y[i].item()
            if count[lbl] < per_class:
                selected_test_idx.append(i.item())
                count[lbl] += 1
            if min(count) == per_class: break
        selected_test_idx = torch.LongTensor(selected_test_idx)
    return selected_test_idx



        
def top1_label_getter(pred_texts, label_names):
    preds = []
    label_names = [l.lower() for l in label_names]
    for i, t in enumerate(pred_texts):
        match = False
        clean_t = t.replace('.', ' ')
        clean_t = clean_t.lower()
        try:
            start = clean_t.find('[')
            end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
            result = ast.literal_eval(list_str)
            res = result[0]
            if res in label_names:
                this = label_names.index(res)
                preds.append(this)
                match = True
            else:
                edits = np.array([editdistance.eval(res, l) for l in label_names])
                this = np.argmin(edits)
                preds.append(this)
                match = True
        except Exception:
            for i, l in enumerate(label_names):
                if l.lower() in clean_t:
                    preds.append(i)
                    match = True
                    break
        if not match:
            edits = np.array([editdistance.eval(clean_t, l) for l in label_names])
            this = np.argmin(edits)
            preds.append(this)

    preds = torch.LongTensor(preds)
    return preds




def annotator(pred_texts, label_names):
    label_names = [l.lower() for l in label_names]
    anno = []
    conf = []
    for i, t in enumerate(pred_texts):
        match = False
        # clean_t = t.replace('.', ' ')
        clean_t = t.lower()
        try:
            start = clean_t.find('{')
            end = clean_t.find('}', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
            # import ipdb; ipdb.set_trace()
            result = ast.literal_eval(list_str)
            # import ipdb; ipdb.set_trace()
            label = ast.literal_eval(result['category'])
            confidence = result['confidence level']
            l = label_names.index(label[0])
            anno.append(l)
            conf.append(confidence)
            # import ipdb; ipdb.set_trace()
        except Exception:
            anno.append(-1)
            conf.append(0)

    anno = torch.LongTensor(anno)
    return anno, conf
