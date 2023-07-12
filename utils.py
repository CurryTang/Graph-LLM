import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import string 
import random
import json
import torch 
import editdistance
import os.path as osp
from torch_geometric.utils import to_networkx, coalesce, to_torch_csr_tensor, to_edge_index, remove_self_loops
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import gc
import os
import time
import datetime
import pytz
import pickle
import errno
import itertools
import tiktoken
import ast


def label_getter(result, label_names):
    l = []
    for line in result:
        find = False 
        for i, label in enumerate(label_names):
            if label.lower() in line:
                l.append(i)
                find = True
                break 
        if not find:
            edits = np.array([editdistance.eval(line, l) for l in label_names])
            l.append(np.argmin(edits))
    return torch.tensor(l)
    


def tsne_2d_plot(embeddings, filename, labels):
    # Apply t-SNE to project embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y)
        plt.text(x+0.01, y+0.01, labels[i], fontsize=9)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("2D t-SNE Visualization of Embeddings")
    plt.show()
    plt.savefig(filename)


# Function to plot tensors using t-SNE with different colors for different labels
def plot_tensors_with_tsne(tensors, labels):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tensors)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(tsne_results[labels == label, 0], tsne_results[labels == label, 1], label=label)

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.title("2D t-SNE Visualization of Tensors with Different Colors for Labels")
    plt.show()




def random_string(length):
    # Choose from all possible characters (letters and digits)
    possible_chars = string.ascii_letters + string.digits
    
    # Generate a random string of the specified length
    result = ''.join(random.choice(possible_chars) for _ in range(length))
    
    return result


def read_jsonl(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            result.append(json_obj)
    return result






def error_analysis(output_file_path, ground_truth, label_names, gnn_output, mlp_output):
    result = read_jsonl(output_file_path)
    pred_texts = [x[1]['choices'][0]['message']['content'] for x in result]
    prompts = [x[0]['messages'][0]['content'] for x in result]
    preds = label_getter(pred_texts, label_names).numpy()
    gtnp = ground_truth.numpy()
    idx = np.arange(len(pred_texts))
    dataset = output_file_path.split('/')[-1].split("_")[0]
    dataset_path = Path(osp.join("./error_analysis", dataset))
    dataset_path.mkdir(parents=True, exist_ok=True)
    ## gpt is correct, mlp is correct
    mlpconfusion_matrix, mlpul, mlpur, mlpll, mlplr = create_confusion_matrix(gtnp, preds, mlp_output)
    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1.2)
    sns.heatmap(mlpconfusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    # Configure the plot
    plt.xlabel("MLP")
    plt.ylabel("GPT")
    plt.title("Ouput Comparison")
    plt.tight_layout()
    plt.savefig(str(dataset_path / "mlp_gpt.png"))
    mlpul_file = dataset_path / "mlpul.txt"
    mlpur_file = dataset_path / "mlpur.txt"
    mlpll_file = dataset_path / "mlpll.txt"
    mlplr_file = dataset_path / "mlplr.txt"
    total = [mlpul_file, mlpur_file, mlpll_file, mlplr_file]
    res = [mlpul, mlpur, mlpll, mlplr]
    for r, f in zip(res, total):
        with f.open("w") as sentinel:
            pass
        for t in r:
            with f.open("a") as ff:
                ff.write(f"Sample {t + 1}\n")
                ff.write(prompts[t])
                ff.write("Ground truth: {}\n".format(label_names[gtnp[t]]))
                ff.write("GPT Prediction: {}\n".format(pred_texts[t]))
                ff.write("MLP Prediction: {}\n".format(label_names[mlp_output[t]]))
    
    
    gnnconfusion_matrix, gnnul, gnnur, gnnll, gnnlr = create_confusion_matrix(gtnp, preds, gnn_output)
    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1.2)
    sns.heatmap(gnnconfusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    # Configure the plot
    plt.xlabel("gnn")
    plt.ylabel("GPT")
    plt.title("Ouput Comparison")
    plt.tight_layout()
    plt.savefig(str(dataset_path / "gnn_gpt.png"))
    gnnul_file = dataset_path / "gnnul.txt"
    gnnur_file = dataset_path / "gnnur.txt"
    gnnll_file = dataset_path / "gnnll.txt"
    gnnlr_file = dataset_path / "gnnlr.txt"
    total = [gnnul_file, gnnur_file, gnnll_file, gnnlr_file]
    res = [gnnul, gnnur, gnnll, gnnlr]
    for r, f in zip(res, total):
        with f.open("w") as sentinel:
            pass
        for t in r:
            with f.open("a") as ff:
                ff.write(f"Sample {t + 1}\n")
                ff.write(prompts[t])
                ff.write("Ground truth: {}\n".format(label_names[gtnp[t]]))
                ff.write("Prediction: {}\n".format(pred_texts[t]))
                ff.write("GNN Prediction: {}\n".format(label_names[gnn_output[t]]))


    return torch.tensor(preds)



def create_confusion_matrix(ground_truth_list, prediction_list_a, prediction_list_b):
    confusion_matrix = np.zeros((2, 2), dtype=int)
    ul, ur, ll, lr = [], [], [], []
    for i, (gt, pred_a, pred_b) in enumerate(zip(ground_truth_list, prediction_list_a, prediction_list_b)):
        if pred_a == gt and pred_b == gt:
            confusion_matrix[0, 0] += 1
            ul.append(i)
        elif pred_a == gt and pred_b != gt:
            confusion_matrix[0, 1] += 1
            ur.append(i)
        elif pred_a != gt and pred_b == gt:
            confusion_matrix[1, 0] += 1
            ll.append(i)
        elif pred_a != gt and pred_b != gt:
            confusion_matrix[1, 1] += 1
            lr.append(i)

    return confusion_matrix, ul, ur, ll, lr


def sample_from_data(data, sample_num, ego_size):
    nx_comm = to_networkx(data, to_undirected=True, remove_self_loops=True) 
    all_idxs = torch.arange(data.x.shape[0])
    test_node_idxs = all_idxs[data.test_masks[0]]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]
    ego_graphs = []
    centers = []
    for center_node_idx in tqdm(sampled_test_node_idxs):
        center_node_idx_int = center_node_idx.item()
        # ego_graph = nx.ego_graph(nx_comm, center_node_idx_int, radius=2, center = True, undirected=True)
        neighbors = list(nx_comm.neighbors(center_node_idx_int))
        subgraph_list = [center_node_idx_int]
        if len(neighbors) > ego_size:
            sample_node_list = random.sample(list(neighbors), ego_size)
            subgraph_list.extend(sample_node_list)
            subgraph_list = list(set(subgraph_list))
        else:
            subgraph_list.extend(neighbors)
        subgraph_list = list(set(subgraph_list))
        selected_context = nx_comm.subgraph(subgraph_list)
        ego_graphs.append(selected_context)
        centers.append(center_node_idx_int)
    # torch.save([ego_graphs, centers], osp.join(path, filename))
    return ego_graphs, centers



def neighbors(edge_index, node_id):
    row, col = edge_index 
    match_idx = torch.where(row == node_id)[0]
    neigh_nodes = col[match_idx]
    return neigh_nodes.tolist()

def get_sampled_nodes(data_obj, sample_num = -1):
    train_mask = data_obj.train_masks[0]
    # val_mask = data_obj.val_masks[0]
    test_mask = data_obj.test_masks[0]
    all_idxs = torch.arange(data_obj.x.shape[0])
    test_node_idxs = all_idxs[test_mask]
    train_node_idxs = all_idxs[train_mask]
    # val_node_idxs = all_idxs[val_mask]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]
    return sampled_test_node_idxs, train_node_idxs


def get_one_hop_neighbors(data_obj, sampled_test_node_idxs, sample_num = -1):
    ## if sample_nodes == -1, all test nodes within test masks will be considered
    neighbor_dict = {}
    for center_node_idx in sampled_test_node_idxs:
        center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = neighbors(data_obj.edge_index, center_node_idx)
    return neighbor_dict

def get_two_hop_neighbors_no_multiplication(data_obj, sampled_test_node_idxs, sample_num = -1):
    neighbor_dict = {}
    # for center_node_idx in sampled_test_node_idxs:
    one_hop_neighbor_dict = get_one_hop_neighbors(data_obj, sampled_test_node_idxs)
    for key, value in one_hop_neighbor_dict.items():
        this_key_neigh = []
        second_hop_neighbor_dict = get_one_hop_neighbors(data_obj, torch.IntTensor(value))
        second_hop_neighbors = set(itertools.chain.from_iterable(second_hop_neighbor_dict.values()))
        second_hop_neighbors.discard(key)
        neighbor_dict[key] = sorted(list(second_hop_neighbors))
    return neighbor_dict






def get_two_hop_neighbors(data_obj, sampled_test_node_idxs, sample_nodes = -1):
    ## if sample_nodes == -1, all test nodes within test masks will be considered
    neighbor_dict = {}
    N = data_obj.x.shape[0]
    edge_index = data_obj.edge_index
    adj = to_torch_csr_tensor(edge_index, size=(N, N))
    edge_index2, _ = to_edge_index(adj @ adj)
    edge_index2, _ = remove_self_loops(edge_index2)
    edge_index = torch.cat([edge_index, edge_index2], dim=1)
    two_hop_edge_index, _ = coalesce(edge_index, None, N)
    for center_node_idx in sampled_test_node_idxs:
        center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = neighbors(two_hop_edge_index, center_node_idx)
    return neighbor_dict









def prompt_with_demonstration(centers, texts, label_names, topk_res, label_to_text, demo_number = 1, strategy = "random", need_cot = False):
    prompts = []
    if centers == None:
        centers = list(range(len(texts)))
    for center in centers:
        background = ""
        if topk_res != None:
            topk = topk_res[center]
        else:
            topk = list(range(len(label_names)))
        topk_str = ", ".join([label_names[i] for i in topk])
        if demo_number > 0:
            for i in topk:
                name = label_names[i]
                demo_list = label_to_text[i]
                choices = random.sample(demo_list, k = demo_number)
                for c in choices:
                    background += "Paper:\n"
                    background += c
                    background += "\n"
                    background += "Which one of the following categories {} is the category of this paper?\n".format(topk_str)
                    background += "Label: "
                    background += name
                    background += "\n"
        background += "Paper:\n"
        background += texts[center] 
        background += "\n"
        background += "Which one of the following categories {} is the category of this paper?\n".format(topk_str)
        background += "Label: " 
        prompts.append(background)
    return prompts


def prompt_paraphrase(texts):
    prompts = []
    for t in texts:
        prompts.append(f"{t}\nParaphrase the paper into a paragraph.")
    return prompts


def generate_topk_using_gpt(texts):
    prompts = []
    for t in texts:
        prompt = "Paper: \n \
         Training algorithms for hidden Markov models using entropy based distance functions. : We present new algorithms for parameter estimation of HMMs. By adapting a framework used for supervised learning, we construct iterative algorithms that maximize the likelihood of the observations while also attempting to stay close to the current estimated parameters. We use a bound on the relative entropy between the two HMMs as a distance measure between them. The result is new iterative training algorithms which are similar to the EM (Baum-Welch) algorithm for training HMMs. The proposed algorithms are composed of a step similar to the expectation step of Baum-Welch and a new update of the parameters which replaces the maximization (re-estimation) step. The algorithm takes only negligibly more time per iteration and an approximated version uses the same expectation step as Baum-Welch. We evaluate experimentally the new algorithms on synthetic and natural speech pronunciation data. For sparse models, i.e. models with relatively small number of non-zero parameters, the proposed algorithms require significantly fewer iterations. \n \
        Which of the following seven categories 'Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods' can be the categories of this paper? Output the most 3 possible options as a python list \n \
        Output: \n \
        ['Theory', 'Case Based', 'Neural Networks']\n"
        prompt += f"Paper: \n \
        {t} \n \
        Which of the following seven categories 'Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods' can be the categories of this paper? Output the most 3 possible options as a python list \n \
        Output: \n"
        prompts.append(prompt)
    return prompts





def entropy(prediction_probs):
    entropy = -torch.sum(prediction_probs * torch.log(prediction_probs), dim=-1)
    return entropy


def norm_entropy(prediction_probs):
    entro = entropy(prediction_probs)
    V = prediction_probs.shape[1]
    return (V * torch.exp(entro) - 1) / (V - 1)


def confidence_analysis(logits, gpt_out, gt, split = 10):
    prob = torch.softmax(logits, dim = -1)
    confidence = norm_entropy(prob)
    new_idx = torch.argsort(confidence, dim = 0)
    model_pred = logits.argmax(dim = -1)
    new_gpt_out = gpt_out[new_idx]
    new_gt = gt[new_idx]
    new_pred = model_pred[new_idx]
    gpt_split = torch.chunk(new_gpt_out, split, dim = 0)
    gt_split = torch.chunk(new_gt, split, dim = 0)
    pred_split = torch.chunk(new_pred, split, dim = 0)
    accs = []
    for i in range(split):
        predictions = gpt_split[i]
        ground_truth = gt_split[i]
        pred = pred_split[i]

        # Calculate accuracy for the current split
        correct = (predictions == ground_truth).sum().item()
        model_correct = (pred == ground_truth).sum().item()
        total = predictions.size(0)
        accuracy = correct / total
        model_accuracy = model_correct / total
        accs.append((model_accuracy, accuracy))

    return accs     


def entity_extraction(texts, instruction = " The extracted terms should be relevant to artificial intelligence, machine learning"):
    prompts = []
    for t in texts:
        prompt = f"You should work like a named entity recognizer. \n Paper: \n {t} \n Extract the technical terms from this paper and output a description for each terms in the format of a python dict, with the format {{'XX': 'XXX', 'YY': 'YYY'}}. {instruction} \n "
        prompts.append(prompt)
    return prompts

def delete_after_brace(s):
    index = s.rfind('}')
    if index == -1:  # '}' not found in the string
        return s
    else:
        return s[:index+1]



def _judge_type(data):
    min_val, max_val = data.min(), data.max()
    _dtype = type(min_val)
    if np.issubdtype(_dtype, np.integer):
        if max_val <= 1 and min_val >= 0:
            _dtype = np._bool
        if max_val <= 255 and min_val >= 0:
            _dtype = np.uint8
        elif max_val <= 65535 and min_val >= 0:
            _dtype = np.uint16
        elif max_val <= 2147483647 and min_val >= -2147483647:
            _dtype = np.int32
    elif np.issubdtype(_dtype, np.float):
        _dtype = np.float16
    return _dtype


def save_memmap(data: np.ndarray, path, dtype=None, node_chunk_size=1000000, log=print):
    # ! Determine the least memory cost type

    dtype = _judge_type(data) if dtype is None else dtype

    # ! Store memory map
    x = np.memmap(path, dtype=dtype, mode='w+',
                  shape=data.shape)

    # for i in tqdm(range(0, data.shape[0], node_chunk_size)):
    for i in range(0, data.shape[0], node_chunk_size):
        j = min(i + node_chunk_size, data.shape[0])
        x[i:j] = data[i:j]
    log(f'Saved {path} as {dtype}...')
    del x
    gc.collect()
    # log('releas x')
    return  # SN(type=dtype, path=path, shape=data.shape)


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper



def compute_loss(logits, labels, loss_func, is_gold=None, pl_weight=0.5, is_augmented=False):
    """
    Combine two types of losses: (1-α)*MLE (CE loss on gold) + α*Pl_loss (CE loss on pseudo labels)
    """
    import torch as th

    if is_augmented and ((n_pseudo := sum(~is_gold)) > 0):
        deal_nan = lambda x: 0 if th.isnan(x) else x
        mle_loss = deal_nan(loss_func(logits[is_gold], labels[is_gold]))
        pl_loss = deal_nan(loss_func(logits[~is_gold], labels[~is_gold]))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = loss_func(logits, labels)
    return loss


def pickle_save(var, f_name):
    mkdir_list([f_name])
    pickle.dump(var, open(f_name, 'wb'))
    print(f'Saved {f_name}')



def run_command_parallel(cmd, gpus, log_func=print):
    _ = cmd.split('python ')
    env_path, variables = _[0], _[1]
    cmd = f'CUDA_VISIBLE_DEVICES={gpus} {env_path}torchrun --master_port={find_free_port()} --nproc_per_node={len(gpus.split(","))} {variables}'
    run_command(cmd, log_func)


def run_command(cmd, log_func=print):
    log_func(f'Running command:\n{cmd}')
    ret_value = os.system(cmd)
    if ret_value != 0:
        raise ValueError(f'Failed to operate {cmd}')


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def mkdir_list(p_list, use_relative_path=True, log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists or a single path

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    p_list = p_list if isinstance(p_list, list) else [p_list]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = get_dir_of_file(p)
        mkdir_p(p, log)


def find_free_port():
    from contextlib import closing
    import socket
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def check_path_dict(path_dict):
    # Check if all paths in path_dict already exists.
    try:
        for k, p in path_dict.items():
            assert os.path.exists(p), f'{k} not found.'
        return True
    except:
        return False


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


def list_dir(dir_name, error_msg=None):
    try:
        f_list = os.listdir(dir_name)
        return f_list
    except FileNotFoundError:
        if error_msg is not None:
            print(f'{error_msg}')
        return []


def silent_remove(file_or_path):
    # Modified from 'https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist'
    import shutil
    try:
        if file_or_path[-1] == '/':
            shutil.rmtree(file_or_path)
        else:
            os.remove(file_or_path)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def remove_file(f_list):
    'Remove file or file list'
    f_list = f_list if isinstance(f_list, list) else [f_list]
    for f_name in f_list:
        silent_remove(f_name)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def get_grand_parent_dir(f_name):
    from pathlib import Path
    if '.' in f_name.split('/')[-1]:  # File
        return get_grand_parent_dir(get_dir_of_file(f_name))
    else:  # Path
        return f'{Path(f_name).parent}/'


def get_abs_path(f_name, style='command_line'):
    if style == 'python':
        cur_path = os.path.abspath(os.path.dirname(__file__))
    elif style == 'command_line':
        cur_path = os.path.abspath(os.path.dirname(__file__)).replace(' ', '\ ')

    root_path = cur_path.split('src')[0]
    return os.path.join(root_path, f_name)


def eval(input_dict):
    y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
    y_true = torch.tensor(y_true).reshape(-1)
    y_pred = torch.tensor(y_pred).reshape(-1)

    acc = torch.sum(y_true == y_pred) / y_true.shape[0]
    return acc


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def delete_non_tensor_attributes(data):
    for attr_name in data.keys:
        if not isinstance(data[attr_name], torch.Tensor):
            delattr(data, attr_name)
    return data


def print_str_to_file(str, filename):
    with open(filename, 'w') as f:
        f.write(str)


def knowledge_augmentation(original_texts: list, augment_knowledge: list, strategy: str = "back"):
    """
        If inplace, insert knowledge into original_texts.
        If back, insert it to back
        If separate, separate two texts, encode each one
    """
    total_texts = []
    total_know = []
    for texts, know_t in zip(original_texts, augment_knowledge):
        know = know_t[0]
        dict_str = know
        know_str = ""
        t = texts
        try:
            first_curly = know.index('{')
            second_curly = know.index('}')
            dict_str = know[first_curly:second_curly+1] 
            know_dict = ast.literal_eval(dict_str)
            for key, value in know_dict.items():
                if strategy == "back":
                    t += f" {value} "
                    know_str = dict_str
                elif strategy == "separate":
                    know_str += f" {value};"
                    # total_know.append(know_str)
                    # total_texts.append(texts)
                elif strategy == "inplace":
                    pos = texts.find(key) + len(key)
                    t = texts[:pos] + f" ({value}) " + texts[pos:]
                    know_str = dict_str
                    # total_texts.append(texts)
        except Exception:
            ## format error
            if strategy == "back" or strategy == "inplace":
                t += dict_str
                # total_texts.append(texts)
                know_str = dict_str
            elif strategy == "separate":
                know_str += dict_str
        
        total_know.append(know_str)
        total_texts.append(t)
    return total_texts, total_know

            


def load_mapping():
    arxiv_mapping = {'arxiv cs ai': 'Artificial Intelligence', 'arxiv cs cl': 'Computation and Language', 'arxiv cs cc': 'Computational Complexity', 'arxiv cs ce': 'Computational Engineering, Finance, and Science', 'arxiv cs cg': 'Computational Geometry', 'arxiv cs gt': 'Computer Science and Game Theory', 'arxiv cs cv': 'Computer Vision and Pattern Recognition', 'arxiv cs cy': 'Computers and Society', 'arxiv cs cr': 'Cryptography and Security', 'arxiv cs ds': 'Data Structures and Algorithms', 'arxiv cs db': 'Databases', 'arxiv cs dl': 'Digital Libraries', 'arxiv cs dm': 'Discrete Mathematics', 'arxiv cs dc': 'Distributed, Parallel, and Cluster Computing', 'arxiv cs et': 'Emerging Technologies', 'arxiv cs fl': 'Formal Languages and Automata Theory', 'arxiv cs gl': 'General Literature', 'arxiv cs gr': 'Graphics', 'arxiv cs ar': 'Hardware Architecture', 'arxiv cs hc': 'Human-Computer Interaction', 'arxiv cs ir': 'Information Retrieval', 'arxiv cs it': 'Information Theory', 'arxiv cs lo': 'Logic in Computer Science', 'arxiv cs lg': 'Machine Learning', 'arxiv cs ms': 'Mathematical Software', 'arxiv cs ma': 'Multiagent Systems', 'arxiv cs mm': 'Multimedia', 'arxiv cs ni': 'Networking and Internet Architecture', 'arxiv cs ne': 'Neural and Evolutionary Computing', 'arxiv cs na': 'Numerical Analysis', 'arxiv cs os': 'Operating Systems', 'arxiv cs oh': 'Other Computer Science', 'arxiv cs pf': 'Performance', 'arxiv cs pl': 'Programming Languages', 'arxiv cs ro': 'Robotics', 'arxiv cs si': 'Social and Information Networks', 'arxiv cs se': 'Software Engineering', 'arxiv cs sd': 'Sound', 'arxiv cs sc': 'Symbolic Computation', 'arxiv cs sy': 'Systems and Control'}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Diabetes Mellitus, Experimental',
        'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
        'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    }
    cora_mapping = {
        'Rule_Learning': "Rule Learning",
        'Neural_Networks': "Neural Networks",
        'Case_Based': "Case Based",
        'Genetic_Algorithms': "Genetic Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement Learning",
        'Probabilistic_Methods': "Probabilistic Methods"
    }

    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}
    return arxiv_mapping, citeseer_mapping, pubmed_mapping, cora_mapping, products_mapping