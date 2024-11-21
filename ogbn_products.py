from datasets import load_dataset
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import torch_geometric.transforms as T
import openai
from tqdm import tqdm
import torch
import time
import pyarrow as pa
import os
import re
from tqdm import tqdm

def set_api_key():
    openai.api_key = "XXX"



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
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def ogb_data(normalize_features = False, transform = None):
    dataset = get_ogbn_dataset("ogbn-products", normalize_features, transform=transform)
    data = dataset[0]
    return data


def compress_embeddings(embedding_list, name = 'ogb_product_features.pt'):
    arr = pa.array(embedding_list)
    torch.save(arr, name)



def get_raw_dataset(raw_train = "raw_data/Amazon-3M.raw/trn.json", raw_test = "raw_data/Amazon-3M.raw/tst.json", 
                    label2cat = "raw_data/ogbn_products/mapping/labelidx2productcategory.csv",
                    idx2asin = "raw_data/ogbn_products/mapping/nodeidx2asin.csv"
                    ):
    train_part = load_dataset("json", data_files=raw_train)
    test_part = load_dataset("json", data_files=raw_test)
    train_df = train_part['train'].to_pandas()
    test_df = test_part['train'].to_pandas()
    combine_df = pd.concat([train_df, test_df], ignore_index=True)
    label2cat_df = pd.read_csv(label2cat)
    idx2asin = pd.read_csv(idx2asin)
    idx_mapping = {row[0]: row[1] for row in idx2asin.values}
    content_mapping = {row[0]: (row[1], row[2]) for row in combine_df.values}
    return idx_mapping, content_mapping


def openai_ada_api(input_list, model_name = 'text-embedding-ada-002', max_len = 8190):
    input_list = [x[:max_len] for x in input_list]
    res = openai.Embedding.create(input = input_list, model=model_name)['data']
    res = [x['embedding'] for x in res]
    return res

def save_large_features(large_list, chunk_num = 10):
    chunk_size = len(large_list) // chunk_num
    for i in tqdm(range(chunk_num)):
        if osp.exists(f'ogbn_product_features_{i}.pt'):
            continue
        part = large_list[i * chunk_size: (i + 1) * chunk_size]
        torch.save(part, f'ogbn_product_features_{i}.pt')


def load_backup(path = "ogb/backup/backup.pt"):
    initial = torch.load(path)
    ogb_path = "ogb/backup"
    scatter_filenames = [osp.join(ogb_path, x) for x in os.listdir(ogb_path) if 'backup_' in x and 'compress' not in x]
    sort_filenames = sorted(scatter_filenames, key=lambda x:int(re.findall(r'\d+', x)[-1]))
    for filename in tqdm(sort_filenames):
        size = int(re.findall(r'\d+', filename)[-1])
        intermediate_file = torch.load(filename)
        initial.extend(intermediate_file)
        assert len(initial) <= size
    return initial




def generate_embeddings(cache_size = 1024):
    if not osp.exists('prompt.pt'):
        idx_mapping, content_mapping = get_raw_dataset()
        idx_mapping_list = idx_mapping.items()
        idx_mapping_list = sorted(idx_mapping_list, key=lambda x:x[0])
        prompt_list = []
        for key, value in idx_mapping_list:
            content = content_mapping[value]
            title, abstract = content
            title = title.strip()
            abstract = abstract.strip()
            prompt = f"{title}: {abstract}"
            prompt_list.append(prompt)
        torch.save(prompt_list, 'prompt.pt')
    else:
        prompt_list = torch.load('prompt.pt')
    
    print("prompt loaded")
    cache_num = 0
    backup_num = 1
    result = []
    total_num = 0
    ogb_products = ogb_data()
    if osp.exists('backup.pt'):
        result = load_backup()
        cache_num = len(result)
        total_num = len(result)
        print("backup loaded")
    while cache_num < len(prompt_list):
        prompt_input = prompt_list[cache_num :cache_num + cache_size]
        if osp.exists(osp.join('ogb', f'backup_{total_num}.pt')):
            res = torch.load(osp.join('ogb', f'backup_{total_num}.pt'))
        else:
            res = openai_ada_api(prompt_input)
        cache_num += cache_size
        total_num += min(cache_size, len(prompt_list) - cache_size)
        print(total_num)
        result.extend(res)
        torch.save(res, osp.join('ogb/backup', f'backup_{total_num}.pt'))
        compress_embeddings(res, osp.join(f'compress_backup_{total_num}.pt'))
        print(f"Current number done: {total_num}")
    save_large_features(result, chunk_num=10)
    compress_embeddings(result)
    assert total_num == ogb_products.x.shape[0]
    
    



def generate_ogb_products_pd_df():
    idx_mapping, content_mapping = get_raw_dataset()
    idx_mapping_list = idx_mapping.items()
    idx_mapping_list = sorted(idx_mapping_list, key=lambda x:x[0])
    titles = []
    contents = []
    for _, value in tqdm(idx_mapping_list):
        content = content_mapping[value]
        title, abstract = content
        title = title.strip()
        abstract = abstract.strip()
        titles.append(title)
        contents.append(abstract)
    df = pd.DataFrame({'title': titles, 'content': contents})
    df.to_csv('ogb_products.csv', index=False)





if __name__ == '__main__':
    generate_ogb_products_pd_df()
