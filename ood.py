import torch 
import numpy as np
import ipdb
import os.path as osp

## change this to your path
GOOD_ARXIV = "/egr/research-dselab/chenzh85/toy_experiments/GOODArxiv"
data_path = "/egr/research-dselab/chenzh85/toy_experiments/ogb/preprocessed_data/new/"

def ood_main():
    arxiv_ood_split_degree_concept = torch.load(osp.join(GOOD_ARXIV, "degree/processed/concept.pt"))
    arxiv_ood_split_degree_concept = arxiv_ood_split_degree_concept[0]
    arxiv_ood_split_degree_covariate = torch.load(osp.join(GOOD_ARXIV, "degree/processed/covariate.pt"))
    arxiv_ood_split_degree_covariate = arxiv_ood_split_degree_covariate[0]

    arxiv_ood_split_time_concept = torch.load(osp.join(GOOD_ARXIV, "time/processed/concept.pt"))
    arxiv_ood_split_time_concept = arxiv_ood_split_time_concept[0]
    arxiv_ood_split_time_covariate = torch.load(osp.join(GOOD_ARXIV, "time/processed/covariate.pt"))
    arxiv_ood_split_time_covariate = arxiv_ood_split_time_covariate[0]

    llm_y = torch.load(osp.join(data_path, "arxiv_fixed_pl.pt"))
    pseudo_labels = llm_y.x[:, 0][:]
    pseudo_labels -= 1

    gt = llm_y.y

    ood_type = ['concept_degree', 'covariate_degree', 'concept_time', 'covariate_time']
    ## evaluate the accuracy according to the environment id
    for i, data in enumerate([arxiv_ood_split_degree_concept, arxiv_ood_split_degree_covariate, arxiv_ood_split_time_concept, arxiv_ood_split_time_covariate]):
        name = ood_type[i]
        env_id_max = data.env_id.max()
        avg = []
        for k in range(env_id_max + 1):
            mask = (data.env_id == k)
            pseudo_label = pseudo_labels[mask]
            gt_label = gt[mask]
            acc = (pseudo_label == gt_label).float().mean()
            print(f"{name} {k} {acc}")
            avg.append(acc)

        mask = data.val_mask
        pseudo_label = pseudo_labels[mask]
        gt_label = gt[mask]
        acc = (pseudo_label == gt_label).float().mean()
        print(f"{name} val {acc}")
        avg.append(acc)
        mask = data.test_mask
        pseudo_label = pseudo_labels[mask]
        gt_label = gt[mask]
        acc = (pseudo_label == gt_label).float().mean()
        print(f"{name} test {acc}")
        avg_acc = np.mean(avg)
        std_acc = np.std(avg)
        print(f"{name} avg {avg_acc} std {std_acc}")
    




if __name__ == '__main__':
    ood_main()
