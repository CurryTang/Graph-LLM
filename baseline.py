from args import *
from data import get_dataset, set_seed_config, set_api_key, pkl_and_write, get_tf_idf_by_texts
import torch
from train_utils import train, test, get_optimizer, confidence_test, topk_test, to_inductive, batch_train, batch_test
from models import get_model
import numpy as np
import ipdb
import optuna
from torch.utils.tensorboard import SummaryWriter
import openai
from copy import deepcopy
import logging
import time
from torch_geometric.utils import index_to_mask
import optuna
import sys
from hyper import hyper_search
import os.path as osp
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from utils import delete_non_tensor_attributes
from ogb.nodeproppred import Evaluator
from collections import defaultdict



def train_pipeline_batch(seeds, args, epoch, data, writer, need_train, mode="main"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_result_acc = []
    early_stop_accum = 0
    val_result_acc = []
    out_res = []
    best_val = 0
    evaluator = Evaluator(name='ogbn-products')
    if args.inductive:
        data = to_inductive(data)
    if mode == "main":
        split_num = args.num_split
    else:
        split_num = args.sweep_split
    split = 0
    data.train_mask = data.train_masks[split]
    data.val_mask = data.val_masks[split]
    data.test_mask = data.test_masks[split]
    data = delete_non_tensor_attributes(data)
    assert split_num == 1
    for seed in seeds:
        set_seed_config(seed)
        model = get_model(args).to(device)
        optimizer, scheduler = get_optimizer(args, model)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_val = 0
        for split in range(split_num):
            if args.normalize:
                data.x = F.normalize(data.x, dim = -1)
            input_nodes = torch.arange(data.x.shape[0])[data.train_mask]
            # import ipdb; ipdb.set_trace()
            data = data.to(device, 'x', 'y')
            subgraph_loader = NeighborLoader(data, input_nodes=input_nodes,
                                num_neighbors=[15, 10, 5],
                    batch_size=1024, shuffle=True,
                    num_workers=4)
            val_loader = NeighborLoader(data, input_nodes=None, batch_size=4096, shuffle=False,
                                num_neighbors=[-1], num_workers=1, persistent_workers=True)
            # import ipdb; ipdb.set_trace()
            for epoch in range(1, args.epochs + 1):
                train_loss = batch_train(model, subgraph_loader, optimizer, device)
                if scheduler:
                    scheduler.step()
                val_acc = batch_test(model, data, evaluator, val_loader, device, data.val_mask)
                print(f"Epoch {epoch}: Train loss: {train_loss}, Val acc: {val_acc}")
                if val_acc > best_val:
                    best_val = val_acc
                    best_model = deepcopy(model)
                    early_stop_accum = 0
                else:
                    if epoch >= args.early_stop_start:
                        early_stop_accum += 1
                    if early_stop_accum > args.early_stopping and epoch >= args.early_stop_start:
                        break
            test_acc = batch_test(model, data, evaluator, val_loader, device, data.test_mask)
            val_result_acc.append(val_acc)
            test_result_acc.append(test_acc)
    return test_result_acc, val_result_acc

            






def train_pipeline(seeds, args, epoch, data, writer, need_train, mode="main"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_result_acc = []
    early_stop_accum = 0
    val_result_acc = []
    out_res = []
    
    if args.inductive:
        data = to_inductive(data)
    if mode == "main":
        split_num = args.num_split
    else:
        split_num = args.sweep_split
    for i, seed in enumerate(seeds):
        best_val = 0
        set_seed_config(seed)
        model = get_model(args).to(device)
        optimizer, scheduler = get_optimizer(args, model)
        loss_fn = torch.nn.CrossEntropyLoss()            # if hasattr(data, "xs"):
            #     data.x = data.xs[0]
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(device)
        data.train_mask = data.train_masks[i]
        data.val_mask = data.val_masks[i]
        data.test_mask = data.test_masks[i]
        if 'ft' in args.data_format:
            data.x = data.xs[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
        if args.split == 'pl_fixed' or args.split == 'pl_random':
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
            data.backup_y = data.y
            data.y = data.ys[i]
        # import ipdb; ipdb.set_trace()
        for i in range(epoch):
            # ipdb.set_trace()
            train_mask = data.train_mask
            val_mask = data.val_mask
            if need_train:
                train_loss, val_loss, val_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask)
                if writer != None:
                    writer.add_scalar('Loss/train', train_loss, i)
                    writer.add_scalar('Loss/val', val_loss, i)
                    writer.add_scalar('Acc/val', val_acc[0], i)
                if scheduler:
                    scheduler.step()
                if args.output_intermediate:
                    print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                if val_acc[0] > best_val:
                    best_val = val_acc[0]
                    best_model = deepcopy(model)
                    early_stop_accum = 0
                else:
                    if i >= args.early_stop_start:
                        early_stop_accum += 1
                    if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                        break
            else:
                best_model = model
        if 'pl' in args.split:
            data.y = data.backup_y
        test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
        test_result_acc.append(test_acc)
        val_result_acc.append(best_val)
        out_res.append(res)
        # del data 
        # del best_model
    return test_result_acc, val_result_acc, out_res





def main(args = None, custom_args = None, save_best = False):
    seeds = [i for i in range(args.seed_num)]
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if custom_args != None:
        args = replace_args_with_dict_values(args, custom_args)
    data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
    seeds = [i for i in range(args.seed_num)]
    best_model = None
    best_val = 0
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_classes'] = data.y.max().item() + 1
    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True
    if not args.batchify and args.ensemble_string == "":
        data.x = data.x.to(torch.float32)
        test_result_acc, val_result_acc, out_res = train_pipeline(seeds, args, epoch, data, writer, need_train)
        mean_test_acc = np.mean(test_result_acc) * 100
        std_test_acc = np.std(test_result_acc) * 100
        print(f"Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
        print("Test acc: {}".format(test_result_acc))
        pkl_and_write(out_res, f'./output/{args.model_name}_{args.dataset}_{args.data_format}.pkl')
    elif args.ensemble_string != "":
        feats = args.ensemble_string.split(";")
        res = []
        sep_test_acc = defaultdict(list)
        labels = data.y
        test_masks = data.test_masks
        for feat in feats:
            vars(args)['data_format'] = feat
            data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
            vars(args)['input_dim'] = data.x.shape[1]
            vars(args)['num_classes'] = data.y.max().item() + 1
            # model = get_model(args).to(device)
            # optimizer, scheduler = get_optimizer(args, model)
            data.x = data.x.to(torch.float32)
            test_result_acc, val_result_acc, out_res = train_pipeline(seeds, args, epoch, data,  writer, need_train)
            res.append(out_res)
            sep_test_acc[feat] = test_result_acc
        for key, value in sep_test_acc.items():
            mean = np.mean(value) * 100
            std = np.std(value) * 100
            print(f"{key}: {mean:.2f} ± {std:.2f}")
        ensemble_input = [[res[i][j] for i in range(len(feats))] for j in range(len(seeds))]
        ensemble_helper(ensemble_input, labels, test_masks)
    else:
        test_result_acc, val_result_acc = train_pipeline_batch(seeds, args, epoch, data, writer, need_train)
        mean_test_acc = np.mean(test_result_acc) * 100.0
        std_test_acc = np.std(test_result_acc) * 100.0
        print(f"Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        print("Test acc: {}".format(test_result_acc))
    if save_best:
        pkl_and_write(args, osp.join("./bestargs", f"{args.model_name}_{args.dataset}_{args.data_format}.pkl"))
    writer.close()


                
def max_trial_callback(study, trial, max_try):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    n_total_complete = len([t for t in study.trials])
    if n_complete >= max_try or n_total_complete >= 2 * max_try:
        study.stop()
        torch.cuda.empty_cache()


def sweep(args = None):
    # test_seeds = [i for i in range(args.seed_num)]
    sweep_seeds = [0, 1, 2, 3, 4]
    ## get default command line args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{args.dataset}_{args.model_name}_{args.data_format}_{args.split}"
    study = optuna.create_study(study_name=study_name, storage=None, direction='maximize', load_if_exists=True)
    param_f = hyper_search
    sweep_round = args.sweep_round
    study.optimize(lambda trial: sweep_run(trial, args, sweep_seeds, param_f, device), catch=(RuntimeError,), n_trials=sweep_round, callbacks=[lambda study, trial: max_trial_callback(study, trial, sweep_round)], show_progress_bar=True, gc_after_trial=True)
    main(args=args, custom_args = study.best_trial.params, save_best = True)
    print(study.best_trial.params)



def sweep_run(trial, args, sweep_seeds, param_f, device):
    params = param_f(trial, args.data_format, args.model_name, args.dataset)    
    args = replace_args_with_dict_values(args, params)
    data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test).to(device)
    best_model = None
    best_val = 0
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_classes'] = data.y.max().item() + 1
    # model = get_model(args).to(device)
    # optimizer, scheduler = get_optimizer(args, model)
    # loss_fn = torch.nn.CrossEntropyLoss()
    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True
    if not args.batchify and args.ensemble_string == "":
        data.x = data.x.to(torch.float32)
        test_result_acc, val_result_acc, out_res = train_pipeline(sweep_seeds, args, epoch, data, None, need_train, mode="sweep")
    elif args.ensemble_string != "":
        feats = args.ensemble_string.split(";")
        res = []
        sep_test_acc = defaultdict(list)
        labels = data.y
        test_masks = data.test_masks
        for feat in feats:
            vars(args)['data_format'] = feat
            data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
            vars(args)['input_dim'] = data.x.shape[1]
            vars(args)['num_classes'] = data.y.max().item() + 1
            # model = get_model(args).to(device)
            # optimizer, scheduler = get_optimizer(args, model)
            data.x = data.x.to(torch.float32)
            test_result_acc, val_result_acc, out_res = train_pipeline(sweep_seeds, args, epoch, data, None, need_train, mode="sweep")
            res.append(out_res)
            sep_test_acc[feat] = test_result_acc
        for key, value in sep_test_acc.items():
            print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")
        ensemble_input = [[res[i][j] for i in range(len(feats))] for j in range(len(sweep_seeds))]
        mean_test_acc, _ = ensemble_helper(ensemble_input, labels, test_masks)
        return mean_test_acc
    else:
        test_result_acc, val_result_acc = train_pipeline_batch(seeds, args, epoch, data,  writer, need_train, mode="sweep")
    mean_test_acc = np.mean(test_result_acc)
    std_test_acc = np.std(test_result_acc)
    print(f"Test Accuracy: {mean_test_acc} ± {std_test_acc}")
    # mean_val_acc = np.mean(val_result_acc)
    # std_val_acc = np.std(val_result_acc)
    # print(f"Val Accuracy: {mean_val_acc} ± {std_val_acc}")
    return mean_test_acc




@torch.no_grad()
def ensemble_helper(logits, labels, test_masks):
    seeds_num = len(logits)
    accs = []
    for i in range(seeds_num):
        test_mask = test_masks[i].cpu()
        this_seed_logits = logits[i]
        avg_logits = sum(this_seed_logits) / len(this_seed_logits)
        pred = torch.argmax(avg_logits, dim=1).cpu()
        labels = labels.cpu()
        acc = torch.sum(pred[test_mask] == labels[test_mask]).item() / len(labels[test_mask])
        accs.append(acc)
    mean_test_acc = np.mean(accs) * 100.0
    std_test_acc = np.std(accs) * 100.0
    print(f"Ensemble Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
    return mean_test_acc, std_test_acc

    


    
if __name__ == '__main__':
    current_time = int(time.time())
    logging.basicConfig(filename='./logs/{}.log'.format(current_time),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    # set_seed_config(42)
    ## get mode: sweep or main
    args = get_command_line_args()    
    set_api_key()
    # param_search()
    if args.mode == "main":
        main(args = args)
    else:
        sweep(args = args)



