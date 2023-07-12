import torch as th
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import os.path as osp
from utils import init_random_state, init_path, eval
from utils import compute_loss
from data import set_seed_config
import numpy as np
import torch.nn.functional as F
from transformers.models.auto import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer import Trainer, TrainingArguments, IntervalStrategy
import argparse
from torch_geometric.utils import mask_to_index
import ipdb
from ogb.nodeproppred import Evaluator
import os
from data import get_dataset
from utils import knowledge_augmentation

### Adapted from GLEM


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        self.loss_func = loss_func
        init_random_state(seed)

    def forward(self, input_ids, attention_mask, labels = None, return_dict = None):
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        # print(f'{sum(is_gold)} gold, {sum(~is_gold)} pseudo')
        # import ipdb; ipdb.set_trace()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.bert_encoder = model

    @th.no_grad()
    def forward(self, **input):
        # Extract outputs from the model
        outputs = self.bert_encoder(**input, output_hidden_states=True)
        emb = outputs['hidden_states'][-1]  # Last layer
        # Use CLS Emb as sentence emb.
        node_cls_emb = emb.permute(1, 0, 2)[0]
        return TokenClassifierOutput(logits=node_cls_emb)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, loss_func, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.feat_shrink = feat_shrink
        self.emb = emb
        self.pred = pred
        self.loss_func = loss_func


    @th.no_grad()
    def forward(self, input_ids, attention_mask, labels = None, return_dict = None, node_id = None):
        # Extract outputs from the model
        batch_nodes = node_id.cpu().numpy()
        outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        emb = outputs['hidden_states'][-1]  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)
        # Save prediction and embeddings to disk (memmap)
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)
        # Output empty to fit the Huggingface trainer pipeline
        # loss = self.loss_func(logits, labels)
        empty = th.zeros((len(node_id), 1)).cuda()
        return TokenClassifierOutput(loss=empty, logits=logits)
    

class Config():
    def __init__(self, args) -> None:
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.seed = args.seed
        self.seed_num = args.seed_num

        self.feat_shrink = args.feat_shrink
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.att_dropout = args.att_dropout
        self.cla_dropout = args.cla_dropout

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.eval_patience = args.eval_patience
        self.grad_acc_steps = args.grad_acc_steps
        self.lr = args.lr

        self.output_dir = args.output_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.label_smoothing = args.label_smoothing
        self.split_id = args.split_id
        self.eq_batch_size = args.eq_batch_size
        self.split = args.split
        self.local_rank = os.getenv('LOCAL_RANK', -1)
        self.n_gpus = args.n_gpus
        self.use_explanation = args.use_explanation
        if self.model_name == 'deberta-large':
            self.hidden_dim = 1024
        else:
            self.hidden_dim = 768


def get_model_name_mapping(model_name):
    mapping = {
        "deberta-base": "microsoft/deberta-base",
        "deberta-large": "microsoft/deberta-large",
        "bert": "bert-base-uncased"
    }
    return mapping[model_name]



class TextDataset(th.utils.data.Dataset):
    def __init__(self, encodings, raw_texts, pyg_data, labels=None):
        self.encodings = encodings
        self.labels = labels
        self.raw_texts = raw_texts
        self.data_obj = pyg_data


    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx].flatten(),
            'attention_mask': self.encodings['attention_mask'][idx].flatten(),
        }
        # ipdb.set_trace()
        ## for inference model to save
        item['node_id'] = idx
        if self.labels != None:
            item["labels"] = self.labels[idx].to(th.long)
        #item["raw_text"] = self.raw_texts[idx]
        return item

    def __len__(self):
        return len(self.raw_texts)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    import evaluate
    metric = evaluate.load("accuracy")
    logits = th.tensor(logits).to('cuda')
    labels = th.tensor(labels).to('cuda')
    predictions = th.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


class LMTrainer():
    def __init__(self, config, data, metrics, loss_func) -> None:
        self.config = config
        set_seed_config(self.config.seed)
        self.name = get_model_name_mapping(self.config.model_name)
        self.total_data = data
        train_steps = self.total_data.x.shape[0] // self.config.batch_size + 1
        eval_steps = self.config.eval_patience // self.config.batch_size
        warmup_step = int(self.config.warmup_epochs * train_steps)
        # total_steps = self.config.epochs * len(self.total_data.raw_text) // self.config.batch_size
        self.n_labels = self.total_data.y.max().item() + 1
        self.training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            do_train=True,
            do_eval=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            learning_rate=self.config.lr,
            weight_decay=self.config.weight_decay,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.config.grad_acc_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 4,
            warmup_steps=warmup_step,
            num_train_epochs=self.config.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True,
            local_rank=self.config.local_rank,
            report_to='none'
        )
        self.loss_func = loss_func
        pretrained_model = AutoModel.from_pretrained(self.name, cache_dir = "/localscratch/czk")
        self.model = BertClassifier(pretrained_model,
                                    n_labels=self.n_labels,
                                    loss_func=self.loss_func,
                                    feat_shrink=self.config.feat_shrink)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=7, cache_dir="/localscratch/czk")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast = False, cache_dir = "/localscratch/czk")
        if 'inp' in self.config.dataset_name:
            prev = self.config.dataset_name.split('_')[0]
            data_obj = th.load(f"./preprocessed_data/new/{prev}_{self.config.split}_know_inp_sb.pt", map_location='cpu')
            texts_inp, _ = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='inplace')
            X = self.tokenizer(texts_inp, padding=True, truncation=True, max_length=512, return_tensors='pt')
        elif 'sep' in self.config.dataset_name:
            prev = self.config.dataset_name.split('_')[0]
            data_obj = th.load(f"./preprocessed_data/new/{prev}_{self.config.split}_know_sep_sb.pt", map_location='cpu')
            texts_inp, knowledge = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='separate')
            X = self.tokenizer(knowledge, padding=True, truncation=True, max_length=512, return_tensors='pt')
        else:    
            if not self.config.use_explanation:
                X = self.tokenizer(self.total_data.raw_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            else:
                explanation = th.load(f"./preprocessed_data/new/{self.config.dataset_name}_explanation.pt")
                X = self.tokenizer(explanation, padding=True, truncation=True, max_length=512, return_tensors='pt')
        self.num_of_nodes = len(self.total_data.raw_texts)
        self.text_dataset = TextDataset(X, self.total_data.raw_texts, self.total_data, self.total_data.y)

        self.train_dataset = th.utils.data.Subset(
            self.text_dataset, mask_to_index(self.total_data.train_mask))
        self.val_dataset = th.utils.data.Subset(
            self.text_dataset, mask_to_index(self.total_data.val_mask))
        self.test_dataset = th.utils.data.Subset(
            self.text_dataset, mask_to_index(self.total_data.test_mask))

        # ipdb.set_trace()
        
        self.trainer = Trainer(
            self.model,
            args = self.training_args, 
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=metrics)


        self.model.config.dropout = self.config.dropout
        self.model.config.attention_dropout = self.config.att_dropout
        self.best_model = None
        self.metrics = metrics
        # self.trainer.train()

    
    def train(self):
        self.trainer.train()
        self.best_model = self.trainer.model
        th.save(self.trainer.model.state_dict(), init_path(osp.join(self.config.checkpoint_dir, f"{self.config.dataset_name}-{self.config.model_name}.pt")))

    def save(self, finetune = True):
        finetune_str = "finetune" if finetune else "no_finetune"
        if not self.config.use_explanation:
            emb_path = osp.join(self.config.output_dir, f"{self.config.dataset_name}_{finetune_str}_{self.config.split}_{self.config.seed}.emb")
            pred_path = osp.join(self.config.output_dir, f"{self.config.dataset_name}_{finetune_str}_{self.config.split}_{self.config.seed}.pred")
        else:
            emb_path = osp.join(self.config.output_dir, f"{self.config.dataset_name}_{finetune_str}_{self.config.split}_{self.config.seed}_exp.emb")
            pred_path = osp.join(self.config.output_dir, f"{self.config.dataset_name}_{finetune_str}_{self.config.split}_{self.config.seed}_exp.pred")
        self.emb = np.memmap(init_path(emb_path), dtype=np.float16, mode='w+',
                             shape=(self.num_of_nodes, self.config.hidden_dim))
        self.pred = np.memmap(init_path(pred_path), dtype=np.float16, mode='w+',
                              shape=(self.num_of_nodes, self.n_labels))
        
        if finetune:
            emb_save_model = BertClaInfModel(self.best_model, self.emb, self.pred, self.loss_func, self.config.feat_shrink)
        else:
            pretrained_model = AutoModel.from_pretrained(self.name, cache_dir = "/localscratch/czk")
            no_ft_model = BertClassifier(pretrained_model,
                                    n_labels=self.n_labels,
                                    loss_func=self.loss_func,
                                    feat_shrink=self.config.feat_shrink)
            emb_save_model = BertClaInfModel(no_ft_model, self.emb, self.pred, self.loss_func, self.config.feat_shrink)
        
        emb_save_model.eval()
        save_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=False,
            do_train=False, 
            do_eval=True,
            per_device_eval_batch_size=self.config.batch_size,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
            local_rank=self.config.local_rank,
            report_to='none'
        )

        saver = Trainer(model=emb_save_model, args=save_args)
        saver.predict(self.text_dataset)

        ## evaluate the output
        mapping = {
            "cora": "cora",
            "pubmed": "pubmed",
            "citeseer": "citeseer",
            'products': "ogbn-products",
            "arxiv": "ogbn-arxiv"
        }
        if "inp" in config.dataset_name or "sep" in config.dataset_name:
            data_name = config.dataset_name.split("_")[0]
        else:
            data_name = config.dataset_name        
        dataset_name = mapping[data_name]
        total_pred = th.tensor(self.pred)
        res = evaluate(total_pred, self.total_data, dataset_name, 0)
        print(res)



def evaluate(total_pred, total_data, dataset_name, split_id = 0):
    total_pred = th.argmax(total_pred, dim=-1)
    train_mask = total_data.train_mask
    val_mask = total_data.val_mask
    test_mask = total_data.test_mask
    train_input_dict = {
        "y_true": total_pred[train_mask].reshape(-1, 1),
        "y_pred": total_data.y[train_mask].reshape(-1, 1)
    }
    val_input_dict = {
        "y_true": total_pred[val_mask].reshape(-1, 1),
        "y_pred": total_data.y[val_mask].reshape(-1, 1)
    }
    test_input_dict = {
        "y_true": total_pred[test_mask].reshape(-1, 1),
        "y_pred": total_data.y[test_mask].reshape(-1, 1)
    }
    if "ogb" in dataset_name:
        evaluator = Evaluator(name = dataset_name)
        train_acc = evaluator.eval(train_input_dict)['acc']
        val_acc = evaluator.eval(val_input_dict)['acc']
        test_acc = evaluator.eval(test_input_dict)['acc']
        res = {
            "train_acc": train_acc.item() if isinstance(train_acc, th.Tensor) else train_acc,
            "val_acc": val_acc.item() if isinstance(val_acc, th.Tensor) else val_acc,
            "test_acc": test_acc.item() if isinstance(test_acc, th.Tensor) else test_acc
        }
    else:
        train_acc = eval(train_input_dict)
        val_acc = eval(val_input_dict)
        test_acc = eval(test_input_dict)
        res = {
            "train_acc": train_acc.item() if isinstance(train_acc, th.Tensor) else train_acc,
            "val_acc": val_acc.item() if isinstance(val_acc, th.Tensor) else val_acc,
            "test_acc": test_acc.item() if isinstance(test_acc, th.Tensor) else test_acc
        }
    return res







def parse_args():
    parser = argparse.ArgumentParser(description='LM training')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--model', type=str, default="deberta-base")
    parser.add_argument('--feat_shrink', type=str, default="")
    ## follow GLEM
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--grad_acc_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=float, default=0.6)
    parser.add_argument('--eval_patience', type=int, default=50000)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--att_dropout', type=float, default=0.1)
    parser.add_argument('--cla_dropout', type=float, default=0.4)
    parser.add_argument("--split", type=str, default="fixed")
    parser.add_argument("--output_dir", type=str, default="./lmoutput")
    parser.add_argument('--checkpoint_dir', type=str, default="./lmcheckpoint")
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--split_id", type=int, default = 0)
    parser.add_argument("--eq_batch_size", type=int, default = 36)
    parser.add_argument("--n_gpus", type=int, default = 1)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--use_explanation", type=int, default=0)
    parser.add_argument("--use_knowledge", type=int, default=0)
    # parser.add_argument("--")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    command_line_args = parse_args()
    num_of_seeds = [i for i in range(command_line_args.seed_num)]
    config = Config(command_line_args)
    if "inp" in config.dataset_name or "sep" in config.dataset_name:
        data_name = config.dataset_name.split("_")[0]
    else:
        data_name = config.dataset_name
    data_obj = get_dataset(config.seed_num, data_name, config.split, "sbert", 0)
    for i in num_of_seeds:
        # import ipdb; ipdb.set_trace()
        n_labels = data_obj.y.max().item() + 1
        data_obj.train_mask = data_obj.train_masks[i]
        data_obj.val_mask = data_obj.val_masks[i]
        data_obj.test_mask = data_obj.test_masks[i]
        # import ipdb; ipdb.set_trace()
        config.seed = i
        loss_func = th.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, reduction='mean')
        # model = BertClassifier(config.model_name, n_labels, loss_func)
        trainer = LMTrainer(config, data_obj, compute_metrics, loss_func)
        trainer.train()
        trainer.save(finetune = True)
        th.cuda.empty_cache()
        # trainer.save(finetune = False)

