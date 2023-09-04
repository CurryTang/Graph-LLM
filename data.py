import torch_geometric.transforms as T
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.utils import index_to_mask, subgraph
import pandas as pd
import pickle as pkl
from models import sbert, mpnet
import random
import numpy as np
import gzip
from api import openai_ada_api, openai_text_api
from copy import deepcopy
from collections import defaultdict
import itertools
import os
import bs4 
import requests
import openai
from torch_sparse.sample import sample_adj
from torch_geometric.typing import Union, Tensor, SparseTensor, Tuple
from torch_geometric.data import Data
import torch_sparse
from scipy.spatial.distance import cdist
from tqdm import tqdm
from InstructorEmbedding import INSTRUCTOR
from mycolorpy import colorlist as mcp
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F
import gensim.downloader
from torch_geometric.datasets import Planetoid
from collections import Counter
from torch_geometric.utils import homophily
from utils import delete_after_brace
import json
from langchain.embeddings import LlamaCppEmbeddings
from ogbn_products import get_raw_dataset
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import google.generativeai as palm
import yaml



OPENAI_OUT = './openai_out'

def load_secret():
    with open('secret.yaml') as f:
        secret = yaml.safe_load(f)
    return secret


def load_arxiv():
    dataframe = pd.read_csv('./preprocessed_data/ogb_arxiv.csv')
    return dataframe


def ptr2index(ptr: Tensor) -> Tensor:
    ind = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return ind.repeat_interleave(ptr[1:] - ptr[:-1])



def to_edge_index(adj: Union[Tensor, SparseTensor]) -> Tuple[Tensor, Tensor]:
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        if value is None:
            value = torch.ones(row.size(0), device=row.device)
        return torch.stack([row, col], dim=0).long(), value

    if adj.layout == torch.sparse_coo:
        return adj.indices().detach().long(), adj.values()

    if adj.layout == torch.sparse_csr:
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    if adj.layout == torch.sparse_csc:
        col = ptr2index(adj.ccol_indices().detach())
        row = adj.row_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")

def set_seed_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def pkl_and_write(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)
    return path

def read_and_unpkl(path):
    with open(path, 'rb') as f:
        res = pkl.load(f)
    return res 




def get_dataset(seed_num, dataset, split, data_format, low_label_test):
    if low_label_test > 0:
        old_mask = False 
    else:
        old_mask = True
    seeds = [i for i in range(seed_num)]
    if 'pl' in split:
        data = torch.load(f"./preprocessed_data/new/{dataset}_random_{data_format}.pt", map_location='cpu')
    else:
        if data_format == 'raw':
            data = Planetoid(f"./data/planetoid", dataset.capitalize())[0]    
        else:
            data = torch.load(f"./preprocessed_data/new/{dataset}_{split}_{data_format}.pt", map_location='cpu')
    if 'pl' in split:
        pl_data = torch.load(f"./preprocessed_data/new/{dataset}_random_pl.pt")
        # import ipdb; ipdb.set_trace()
        pseudo_labels = pl_data.x[:, 0][:]
        pseudo_labels -= 1
        pl_list = pseudo_labels.tolist()
        # import ipdb; ipdb.set_trace()
        ## TAPE use a different label index with us, so we have to make a transform here
        if dataset == 'cora':
            mapping = {0: 2, 1:3, 2:1, 3:6, 4:5, 5:0, 6:4}
            pl_list = [mapping[i] for i in pl_list]
            pseudo_labels = torch.tensor(pl_list)
    else:
        pl_data = None
    if dataset == "products" or dataset == "arxiv":
        data.train_masks = [data.train_masks[0] for _ in range(seed_num)]
        data.val_masks = [data.val_masks[0] for _ in range(seed_num)]
        data.test_masks = [data.test_masks[0] for _ in range(seed_num)]
        return data
    if old_mask:
        data.train_masks = [data.train_masks[i] for i in range(seed_num)]
        data.val_masks = [data.val_masks[i] for i in range(seed_num)]
        data.test_masks = [data.test_masks[i] for i in range(seed_num)]
        return data
    new_train_masks = []
    new_val_masks = []
    new_test_masks = []
    ys = []
    # generate new masks here
    for s in seeds:
        set_seed_config(s)
        if split == 'fixed':
            ## 20 per class
            fixed_split = LabelPerClassSplit(num_labels_per_class=20, num_valid = 500, num_test=1000)
            t_mask, val_mask, te_mask = fixed_split(data, data.x.shape[0])
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(te_mask)
        elif split == 'pl_fixed':
            total_num = data.x.shape[0]
            test_num = int(total_num * 0.2)
            num_classes = data.y.max().item() + 1
            num_valid = 5 * num_classes 
            fixed_split = LabelPerClassSplit(num_labels_per_class=15, num_valid = num_valid, num_test=total_num)
            y_copy = torch.tensor(data.y)
            t_mask, val_mask, te_mask = fixed_split(data, data.x.shape[0])
            y_copy[t_mask] = pseudo_labels[t_mask]
            y_copy[val_mask] = pseudo_labels[val_mask]
            y_copy[~(t_mask | val_mask | te_mask)] = -1
            # import ipdb; ipdb.set_trace()
            ys.append(y_copy)
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(te_mask)
        elif split == 'pl_random':
            num_classes = data.y.max().item() + 1
            total_num = data.x.shape[0]
            total_label_num = 20 * num_classes
            train_num = 20 * num_classes * 3 // 4
            val_num = total_label_num - train_num
            test_num = int(total_num * 0.2)
            y_copy = torch.tensor(data.y)
            t_mask, val_mask, test_mask = generate_random_mask(data.x.shape[0], train_num, val_num, test_num)
            y_copy[t_mask] = pseudo_labels[t_mask]
            y_copy[val_mask] = pseudo_labels[val_mask]
            y_copy[~(t_mask | val_mask | test_mask)] = -1
            ys.append(y_copy)
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(test_mask)
        else:
            total_num = data.x.shape[0]
            train_num = int(0.6 * total_num)
            val_num = int(0.2 * total_num)
            t_mask, val_mask, te_mask = generate_random_mask(data.x.shape[0], train_num, val_num)
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(te_mask)
        ## start from 
    data.train_masks = new_train_masks
    data.val_masks = new_val_masks
    data.test_masks = new_test_masks
    # import ipdb; ipdb.set_trace()
    if 'pl' in split:
        data.ys = ys
    total_indexes = torch.arange(data.x.shape[0])
    num_of_class = data.y.max().item() + 1
    if low_label_test > 0:
        new_train_masks = []
        new_val_masks = []
        low_label_split = LabelPerClassSplit(num_labels_per_class=low_label_test, num_valid=low_label_test * num_of_class, inside_old_mask=True)
        for i in range(seed_num):
            t_mask = []
            set_seed_config(i)
            train_mask = data.train_masks[i]
            train_idx = total_indexes[train_mask]
            new_train_mask, new_val_mask, _ = low_label_split(data, data.x.shape[0])
            new_train_masks.append(new_train_mask)
            new_val_masks.append(new_val_mask)
        data.train_masks = new_train_masks
        data.val_masks = new_val_masks
    return data


            




class LabelPerClassSplit:
    def __init__(
            self,
            num_labels_per_class: int = 20,
            num_valid: int = 500,
            num_test: int = -1,
            inside_old_mask: bool = False
    ):
        self.num_labels_per_class = num_labels_per_class
        self.num_valid = num_valid
        self.num_test = num_test
        self.inside_old_mask = inside_old_mask

    def __call__(self, data, total_num):
        new_train_mask = torch.zeros(total_num, dtype=torch.bool)
        new_val_mask = torch.zeros(total_num, dtype=torch.bool)
        new_test_mask = torch.zeros(total_num, dtype=torch.bool)

        if self.inside_old_mask:
            old_train_mask = data.train_masks[0]
            old_val_mask = data.val_masks[0]
            old_test_mask = data.test_masks[0]
            perm = torch.randperm(total_num)
            train_cnt = np.zeros(data.y.max().item() + 1, dtype=np.int)

            for i in range(perm.numel()):
                label = data.y[perm[i]]
                if train_cnt[label] < self.num_labels_per_class and old_train_mask[perm[i]].item():
                    train_cnt[label] += 1
                    new_train_mask[perm[i]] = 1
                elif new_val_mask.sum() < self.num_valid and old_val_mask[perm[i]].item():
                    new_val_mask[perm[i]] = 1
                else:
                    if self.num_test != -1:
                        if new_test_mask.sum() < self.num_test and old_test_mask[perm[i]].item():
                            new_test_mask[perm[i]] = 1
                    else:
                        new_test_mask[perm[i]] = 1

            
            return new_train_mask, new_val_mask, new_test_mask
        else:
            perm = torch.randperm(total_num)
            train_cnt = np.zeros(data.y.max().item() + 1, dtype=np.int32)

            for i in range(perm.numel()):
                label = data.y[perm[i]]
                if train_cnt[label] < self.num_labels_per_class:
                    train_cnt[label] += 1
                    new_train_mask[perm[i]] = 1
                elif new_val_mask.sum() < self.num_valid:
                    new_val_mask[perm[i]] = 1
                else:
                    if self.num_test != -1:
                        if new_test_mask.sum() < self.num_test:
                            new_test_mask[perm[i]] = 1
                        else:
                            new_test_mask[perm[i]] = 1

            
            return new_train_mask, new_val_mask, new_test_mask


def get_transform(normalize_features, transform):
    # import ipdb; ipdb.set_trace()
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform
    return transform


def get_ogbn_dataset(name, normalize_features=True, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    dataset = PygNodePropPredDataset(name, path)
    transform = T.Compose([T.ToUndirected()])
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def generate_random_mask(total_node_number, train_num, val_num, test_num = -1):
    random_index = torch.randperm(total_node_number)
    train_index = random_index[:train_num]
    val_index = random_index[train_num:train_num + val_num]
    if test_num == -1:
        test_index = random_index[train_num + val_num:]
    else:
        test_index = random_index[train_num + val_num: train_num + val_num + test_num]
    return index_to_mask(train_index, total_node_number), index_to_mask(val_index, total_node_number), index_to_mask(test_index, total_node_number)









# def get_label_cat_mapping(file_path = 'data/ogbn-arxiv/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'):
#     with gzip.open(file_path, 'rt', encoding='utf-8') as file:
#         df = pd.read_csv(file)
#     rows = df.values
#     mapping = {row[1]: row[0] for row in rows}
#     return mapping, df['arxiv category'].values.tolist()

# def get_paper_id_mapping(file_path = 'data/ogbn-arxiv/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz'):
#     with gzip.open(file_path, 'rt', encoding='utf-8') as file:
#         df = pd.read_csv(file)
#     rows = df.values 
#     mapping = {row[1]: row[0] for row in rows}
#     return mapping, df['paper id'].values.tolist()





def generate_instruction_xl_embedding(instruction, sentences, device):
    pairs = []
    for sent in sentences:
        pair = [instruction, sent]
        pairs.append(pair)
    model = INSTRUCTOR('hkunlp/instructor-xl', cache_folder='/tmp', device=device)
    embeddings = model.encode(pairs, batch_size=32, show_progress_bar=True)
    embeddings = torch.FloatTensor(embeddings).to(device)
    return embeddings




def plot_graph_with_cat(G, v, cmap="plasma", node_size=100, output_name = './figures/network.png'):
    vColor =mcp.gen_color_normalized(cmap,data_arr=v + 100)
    pos = nx.spring_layout(G,seed=32)  # Seed layout for reproducibility
    # pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, node_color=vColor, node_size=node_size, width=4, with_labels = False)
    plt.savefig(output_name)
    # return pos
    plt.clf()




def get_tf_idf_by_texts(texts, known_mask, test_mask, max_features = 1433, use_tokenizer = False):
    if known_mask == None and test_mask == None:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = tf_idf_vec.fit_transform(texts)
        torch_feat = torch.FloatTensor(X.todense())
        norm_torch_feat = F.normalize(torch_feat, dim = -1)
        return torch_feat, norm_torch_feat
    if use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir = "/tmp")
        tf_idf_vec = TfidfVectorizer(analyzer="word", max_features=500, tokenizer=lambda x: tokenizer.tokenize(x, max_length=512, truncation=True))
        text_known = texts[known_mask]
        text_test = texts[test_mask]
    else:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        text_known = texts[known_mask]
        text_test = texts[test_mask]
    x_known = tf_idf_vec.fit_transform(text_known)
    x_test = tf_idf_vec.transform(text_test)
    x_known = torch.FloatTensor(x_known.todense())
    x_test = torch.FloatTensor(x_test.todense())
    dim = x_known.shape[1]
    torch_feat = torch.zeros(len(texts), dim)
    torch_feat[known_mask] = x_known 
    torch_feat[test_mask] = x_test
    norm_torch_feat = F.normalize(torch_feat, dim = -1)
    return torch_feat, norm_torch_feat




def get_word2vec(raw_texts):
    raw_text = [[ x for x in line.lower().split(' ') if x.isalpha()] for line in raw_texts]
    w2v_path = load_secret()['word2vec']['path']
    word2vec = KeyedVectors.load_word2vec_format(w2v_path, binary = True)
    vecs = []
    for sentence in raw_text:
        tokens = [x for x in sentence if x.isalpha()]
        word_vectors = [word2vec[word] for word in tokens if word2vec.key_to_index.get(word, None)]
        if len(word_vectors) == 0:
            vecs.append(np.zeros(300))
        else:
            sentence_vectors = np.mean(word_vectors, axis = 0)
            vecs.append(sentence_vectors)
    vecs = np.vstack(vecs)
    vecs = torch.FloatTensor(vecs)
    return vecs



# def ogb_arxiv_dataset():
#     arxiv_df = pd.read_csv("./preprocessed_data/ogb_arxiv.csv")
#     data = ogb_data(False, None).cpu()
#     data.paper_id = arxiv_df['id'].tolist()
#     data.title = arxiv_df['title'].tolist()
#     data.abs = arxiv_df['abstract'].tolist()
#     data.category_names = arxiv_df['category_name'].tolist()
#     data.raw_texts = [x + " " + y for x, y in zip(data.title, data.abs)]
#     _, mapping = get_label_cat_mapping('./data/ogbn-arxiv/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz')
#     data.label_names = mapping
#     torch.save(data, "./preprocessed_data/arxiv_public_w2v.pt")
#     return data


# def ogb_products_dataset():
#     dataset = get_ogbn_dataset("ogbn-products", False, transform=None)
#     data = dataset[0]
#     idx_mapping, content_mapping = get_raw_dataset()
#     idx_mapping_list = idx_mapping.items()
#     idx_mapping_list = sorted(idx_mapping_list, key=lambda x:x[0])
#     prompt_list = []
#     for _, value in idx_mapping_list:
#         content = content_mapping[value]
#         title, abstract = content
#         title = title.strip()
#         abstract = abstract.strip()
#         prompt = f"{title} {abstract}"
#         prompt_list.append(prompt)
#     data.raw_texts = prompt_list
#     label2cat = "./raw_data/ogbn_products/mapping/labelidx2productcategory.csv"
#     label2cat_df = pd.read_csv(label2cat)
#     label_names = label2cat_df['product category'].to_list()
#     label_names = [x if not pd.isna(x) else f"label {i + 1}" for i, x in enumerate(label_names)]
#     data.label_names = label_names 
#     data.category_names = [label_names[i] for i in data.y.reshape(-1).tolist()]
#     torch.save(data, "./ogb/preprocessed_data/products_public_bow.pt")
#     return data


def parse_pubmed(path):
    n_nodes = 19717
    n_features = 500
    n_classes = 3

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = np.zeros((n_nodes, n_classes), dtype='int32')

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path + 'Pubmed-Diabetes.NODE.paper.tab','r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i,line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - 1  # subtract 1 to zero-count
            data_Y[i,label] = 1.

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')
    row = []
    col = []

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab','r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i,line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail],paper_to_index[head]] = 1.0
            data_A[paper_to_index[head],paper_to_index[tail]] = 1.0
            if head != tail:
                row.append(paper_to_index[head]) 
                col.append(paper_to_index[tail])
                row.append(paper_to_index[tail]) 
                col.append(paper_to_index[head])
    
    edge_index = torch.tensor([row, col])

    return data_A, data_X, data_Y, edge_index



def pubmed_to_graph(path, split = "fixed", embedding_type = "original"):
    _, data_X, data_Y, data_edges = parse_pubmed(path)
    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    data = Data()
    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)
    data.y = data.y.argmax(dim = -1)

    if split == "fixed":
        old_data = Planetoid("./raw_data", "PubMed", transform=T.NormalizeFeatures())
        old_data = old_data[0]

        data.train_masks = [old_data.train_mask]
        data.val_masks = [old_data.val_mask]
        data.test_masks = [old_data.test_mask]
    else:
        # node_idx = torch.arange(data.x.shape[0])
        total_num = data.x.shape[0]
        train_mask, val_mask, test_mask = generate_random_mask(total_num, int(total_num * 0.6), int(total_num * 0.2))
        data.train_masks = [train_mask]
        data.val_masks = [val_mask]
        data.test_masks = [test_mask]

    with open('./raw_data/Pubmed-Diabetes/pubmed.json') as f:
        pubmed = json.load(f)
        df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        text.append(t)
    data.raw_texts = text
    if embedding_type == 'sbert':
        sbert_model = sbert('cuda')
        sbert_embeds = sbert_model.encode(data.raw_texts, batch_size=8, show_progress_bar=True)
        data.x = torch.tensor(sbert_embeds)
    data.label_names = ["Diabetes Mellitus, Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"]
    data.category_names = [data.label_names[x] for x in data.y.tolist()]
    torch.save(data, f"./preprocessed_data/pubmed_{split}_{embedding_type}.pt")
    return data


def citeseer_to_graph(citeseer_path = './raw_data/CiteSeer-Orig', split = "public", embedding_type = "original"):
    data = Data()
    citeseer_content = osp.join(citeseer_path, "citeseer_texts.txt")
    citeseer_relation = osp.join(citeseer_path, "citeseer.cites")
    idx_to_row_mapping = {}
    category_name = []
    texts = []
    total_num = 0
    l_names = []
    current_l = 0
    l_mapping = {}
    data_y = []
    with open(citeseer_content, "r") as f:
        while True:
            lines = [f.readline().strip() for _ in range(3)]  # Read three lines
            if not any(lines):  # If all lines are empty, end of file reached
                break
            idx_name = lines[0]
            text = lines[1]
            label_name = lines[2]
            texts.append(text)
            idx_to_row_mapping[idx_name] = total_num
            category_name.append(label_name)
            if l_mapping.get(label_name, None) == None:
                l_mapping[label_name] = current_l
                l_names.append(label_name)
                current_l += 1
            data_y.append(l_mapping.get(label_name))
            total_num += 1
    data.y = torch.tensor(data_y)
    row = []
    col = []
    with open(citeseer_relation, "r") as f:
        for line in f.readlines():
            tup = line.split()
            head, tail = tup[0], tup[1]
            head = idx_to_row_mapping.get(head, None)
            tail = idx_to_row_mapping.get(tail, None)
            if head == None or tail == None:
                continue
            if head != tail:
                row.append(head)
                col.append(head)
                row.append(tail)
                col.append(tail)
    data.edge_index = torch.tensor([row, col])    
    data.raw_texts = texts 
    data.category_names = category_name
    data.label_names = l_names
    if embedding_type == 'sbert':
        sbert_model = sbert('cuda')
        sbert_embeds = sbert_model.encode(data.raw_texts, batch_size=8, show_progress_bar=True)
        data.x = torch.tensor(sbert_embeds)
    else:
        embeds, _ = get_tf_idf_by_texts(texts, max_features=3703)
        data.x = embeds
    if split != "public":
        total_num = data.x.shape[0]
        train_mask, val_mask, test_mask = generate_random_mask(total_num, int(total_num * 0.6), int(total_num * 0.2))
        data.train_masks = [train_mask]
        data.val_masks = [val_mask]
        data.test_masks = [test_mask]
    else:
        spliter = LabelPerClassSplit()
        train_mask, val_mask, test_mask = spliter(data, total_num=total_num)
        data.train_masks = [train_mask]
        data.val_masks = [val_mask]
        data.test_masks = [test_mask]
    torch.save(data, f"./preprocessed_data/citeseer_{split}_{embedding_type}.pt")
    return data


def topk_result(logits, label_names, k = 3):
    output = logits
    topk_res = torch.topk(output, k = 3, dim = -1).indices.to('cpu')
    topk = []
    for _, l in enumerate(topk_res):
        res = []
        for ele in l:
            res.append(label_names[ele].replace('_', ' ').replace('.', ' ').lower())
        topk.append(res)
    return topk

def topk_result_label(logits, k = 3):
    output = logits
    topk_res = torch.topk(output, k = 3, dim = -1).indices.to('cpu')
    topk = []
    for _, l in enumerate(topk_res):
        res = []
        for ele in l:
            res.append(ele.item())
        topk.append(res)
    return topk



def set_api_key():
    openai_secret = load_secret()['openai']['secret']
    openai.api_key = openai_secret




def graph_dataset_statistics(pyg_data):
    label_list = pyg_data.y.tolist()
    counter = Counter(label_list)
    print("####LABEL DISTRIBUTION####")
    for label, count in counter.items():
        print(f"{label}: {count}")
    homophily_ratio = homophily(pyg_data.edge_index, pyg_data.y)
    print(f"Homophily: {homophily_ratio}")


def cora_entity_enhancement():
    data = cora_original_split()
    data.x = torch.tensor(data.x)
    entity_extraction_output = torch.load("cora20_entity.pt", map_location='cpu')
    encoder = sbert("cuda")
    format_error = 0
    embeddings = []
    for line in tqdm(entity_extraction_output):
        new_line = line[0].replace('\n', '')
        new_line = delete_after_brace(new_line)
        error = False
        try:
            line_dict = eval(new_line)
            if not isinstance(line_dict, dict):
                error = True
        except SyntaxError:
            format_error += 1 
            error = True
        if not error:
            descriptions = list(line_dict.values())
            line_embedding = encoder.encode(descriptions, batch_size=8, show_progress_bar=False)
            ## we first try a simple strategy, which averages the embedding
            line_embedding = torch.from_numpy(line_embedding)
            mean_line_embedding = line_embedding.mean(dim = 0)
            embeddings.append(mean_line_embedding)
        else:
            mean_line_embedding = torch.zeros_like(embeddings[0])
            embeddings.append(mean_line_embedding)
    final_embedding = torch.stack(embeddings)
    # final_embedding = final_embedding.mean(dim = 0)
    torch.save(final_embedding, "cora20_entity_embedding.pt")
    print(format_error)
    return final_embedding


def get_llama_embedding(texts):
    embeddings = []
    model_path = load_secret()['llama']['path']
    llama = LlamaCppEmbeddings(model_path=model_path)
    for text in tqdm(texts):
        emb = llama.embed_query(text)
        embeddings.append(torch.tensor(emb))
    return torch.stack(embeddings)


def get_sbert_embedding(texts):
    sbert_model = sbert('cuda')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def batched_data(inputs, batch_size):
    return [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]

def get_e5_large_embedding(texts, device, dataset_name = 'cora', batch_size = 64):
    output_path = osp.join(OPENAI_OUT, dataset_name + "_e5_embedding.pt")
    if osp.exists(output_path):
        return torch.load(output_path, map_location='cpu')
    texts = ["query: " + x for x in texts]
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2', cache_dir='/tmp')
    model = AutoModel.from_pretrained('intfloat/e5-large-v2', cache_dir='/tmp').to(device)
    # Tokenize the input texts
    output = []
    with torch.no_grad():
        for batch in tqdm(batched_data(texts, batch_size)):
            batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            output.append(embeddings.cpu())
            del batch_dict
    output = torch.cat(output, dim = 0)
    torch.save(output, output_path)
    return output


def get_google_embedding(texts, dataset_name = 'cora'):
    output_path = osp.join(OPENAI_OUT, dataset_name + "_google_embedding.pt")
    if osp.exists(output_path):
        return torch.load(output_path, map_location='cpu')
    embeddings = []
    for text in tqdm(texts):
        model = "models/embedding-gecko-001"
        google_api_key = load_secret()['google']['secret']
        palm.configure(api_key=google_api_key)
        embedding_x = palm.generate_embeddings(model=model, text=text)['embedding']
        embeddings.append(torch.tensor(embedding_x))
    embeddings = torch.cat(embeddings, dim = 0)
    torch.save(embeddings, output_path)
    return embeddings

    


    



if __name__ == '__main__':
    #pubmed_to_graph("./raw_data/Pubmed-Diabetes/data/", split="fixed", embedding_type="original")
    citeseer_to_graph("./raw_data/CiteSeer-Orig/", split="public", embedding_type="original")
    #pubmed_to_graph("./raw_data/Pubmed-Diabetes/data/", split="fixed", embedding_type="sbert")
    #citeseer_to_graph("./raw_data/CiteSeer-Orig/", split="public", embedding_type="sbert")
    #pubmed_to_graph("./raw_data/Pubmed-Diabetes/data/", split="random", embedding_type="original")
    citeseer_to_graph("./raw_data/CiteSeer-Orig/", split="random", embedding_type="original")
    #pubmed_to_graph("./raw_data/Pubmed-Diabetes/data/", split="random", embedding_type="sbert")
    #citeseer_to_graph("./raw_data/CiteSeer-Orig/", split="random", embedding_type="sbert")
    # ogb_arxiv_dataset()
    # ogb_products_dataset()
    # get_word2vec(["machine learning"])
