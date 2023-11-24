from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GCNConv, GINConv, SAGEConv
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch_geometric.nn import LabelPropagation
from torch_geometric.nn.models import GAT
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PYGGATConv
import rev.memgcn as memgcn
from rev.rev_layer import SharedDropout
import copy
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
import numpy as np
import math
import tqdm
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair



BIG_CONSTANT = 1e8


def get_model(args):
    if args.model_name == 'MLP':
        return UniversalMLP(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm, args.return_embeds)
    elif args.model_name == 'GCN':
        return GCN(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm)
    elif args.model_name == 'MLP2':
        return DeepMLP(args.input_dim, args.num_classes)
    elif args.model_name == 'LP':
        return LP(args.num_layers, args.alpha)
    elif args.model_name == 'SAGE':
        return SAGE(args.input_dim, args.hidden_dimension, args.num_classes, args.num_layers, args.dropout)
    elif args.model_name == 'GAT':
        return GAT2(args.input_dim, args.hidden_dimension, args.num_layers, args.num_classes, args.dropout, args.dropout, args.num_of_heads, args.num_of_out_heads, args.norm)


class GAT2(torch.nn.Module):
    def __init__(self, num_feat, hidden_dimension, num_layers, num_class, dropout, attn_drop, num_of_heads = 1, num_of_out_heads = 1, norm = None):
        super().__init__()
        self.layers = []
        self.bns = []
        if num_layers == 1:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = False, dropout=attn_drop)
        else:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = True, dropout=attn_drop)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))
        self.layers.append(self.conv1)
        for _ in range(num_layers - 2):
            self.layers.append(
                PYGGATConv(hidden_dimension * num_of_heads, hidden_dimension, num_of_heads, concat = True, dropout = dropout)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        if num_layers > 1:
            self.layers.append(PYGGATConv(hidden_dimension * num_of_heads, num_class, heads=num_of_out_heads,
                             concat=False, dropout=attn_drop).cuda())
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList(self.bns)
        self.norm = norm 
        self.num_layers = num_layers
        self.with_bn = True if self.norm == 'BatchNorm' else False
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.elu(x)
        return x



class GATWrapper(torch.nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, out_size, dropout):
        super().__init__()
        self.gat = GAT(in_size, hidden_size, num_layers, out_size, dropout)
    
    def forward(self, data):
        x, edge_index= data.x, data.edge_index
        return self.gat(x, edge_index)



class UniversalMLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None, return_embeds = False) -> None:
        super().__init__()
        hidden_dimensions = [hidden_dimension] * (num_layers - 1)
        self.hidden_dimensions = [input_dim] + hidden_dimensions + [num_classes]
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=dropout, norm=norm)
        self.return_embeds = False

    def forward(self, data):
        x = data.x
        return self.mlp(x)
    
    def inference(self, x_all, subgraph_loader, device):
        xs = []
        for batch in tqdm.tqdm(subgraph_loader):
            edge_index, n_id, size = batch.edge_index, batch.n_id, batch.batch_size
            edge_index = edge_index.to(device)
            # import ipdb; ipdb.set_trace()
            x = x_all[n_id][:batch.batch_size].to(device)
            x = self.mlp(x)
            xs.append(x.cpu())
        x_all = torch.cat(xs, dim=0)
        return x_all


class DeepMLP(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_size, 1024),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(1024),
                nn.Linear(1024, 512),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(512),
                nn.Linear(512, out_size),
        )
    
    def forward(self, data):
        x = data.x
        return self.mlp(x)




class GCN(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                if norm:
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x



class LP(torch.nn.Module):
    def __init__(self, num_layers, alpha) -> None:
        super().__init__()
        self.lp = LabelPropagation(num_layers, alpha)
    
    def forward(self, data):
        y= data.y
        train_mask = data.train_mask
        return self.lp(y, data.adj_t, train_mask)


def sbert(device):
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/localscratch/czk/huggingface', device=device).to(device)
    return model 


def mpnet(device):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/localscratch/czk/huggingface', device=device).to(device)
    return model 



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = None
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def inference(self, x_all, subgraph_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                edge_index, n_id, size = batch.edge_index, batch.n_id, batch.batch_size
                edge_index = edge_index.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all




class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        assert feat_drop == 0.0 # not implemented
        self.attn_drop = nn.Dropout(attn_drop)
        assert attn_drop == 0.0 # not implemented
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, perm=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                if perm is None:
                    perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst


class RevGATBlock(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        edge_emb,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
    ):
        super(RevGATBlock, self).__init__()

        self.norm = nn.BatchNorm1d(n_heads * out_feats)
        self.conv = GATConv(
                        node_feats,
                        out_feats,
                        num_heads=n_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        negative_slope=negative_slope,
                        residual=residual,
                        activation=activation,
                        use_attn_dst=use_attn_dst,
                        allow_zero_in_degree=allow_zero_in_degree,
                        use_symmetric_norm=use_symmetric_norm,
                    )
        self.dropout = SharedDropout()
        if edge_emb > 0:
            self.edge_encoder = nn.Linear(edge_feats, edge_emb)
        else:
            self.edge_encoder = None

    def forward(self, x, graph, dropout_mask=None, perm=None, efeat=None):
        if perm is not None:
            perm = perm.squeeze()
        out = self.norm(x)
        out = F.relu(out, inplace=True)
        if isinstance(self.dropout, SharedDropout):
            self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)

        if self.edge_encoder is not None:
            if efeat is None:
                efeat = graph.edata["feat"]
            efeat_emb = self.edge_encoder(efeat)
            efeat_emb = F.relu(efeat_emb, inplace=True)
        else:
            efeat_emb = None

        out = self.conv(graph, out, perm).flatten(1, -1)
        return out


class RevGAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
        group=2,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads
        self.group = group

        self.convs = nn.ModuleList()
        self.norm = nn.BatchNorm1d(n_heads * n_hidden)

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            if i == 0 or i == n_layers -1:
                self.convs.append(
                    GATConv(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        use_attn_dst=use_attn_dst,
                        use_symmetric_norm=use_symmetric_norm,
                        residual=True,
                    )
                )
            else:
                Fms = nn.ModuleList()
                fm = RevGATBlock(
                    in_hidden // group,
                    0,
                    0,
                    out_hidden // group,
                    n_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
                for i in range(self.group):
                    if i == 0:
                        Fms.append(fm)
                    else:
                        Fms.append(copy.deepcopy(fm))

                invertible_module = memgcn.GroupAdditiveCoupling(Fms,
                                                                 group=self.group)

                conv = memgcn.InvertibleModuleWrapper(fn=invertible_module,
                                                      keep_input=False)

                self.convs.append(conv)

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = dropout
        self.dp_last = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        self.perms = []
        for i in range(self.n_layers):
            perm = torch.randperm(graph.number_of_edges(),
                                  device=graph.device)
            self.perms.append(perm)

        h = self.convs[0](graph, h, self.perms[0]).flatten(1, -1)

        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)

        for i in range(1, self.n_layers-1):
            graph.requires_grad = False
            perm = torch.stack([self.perms[i]]*self.group, dim=1)
            h = self.convs[i](h, graph, mask, perm)

        h = self.norm(h)
        h = self.activation(h, inplace=True)
        h = self.dp_last(h)
        h = self.convs[-1](graph, h, self.perms[-1])

        h = h.mean(1)
        h = self.bias_last(h)

        return h
               
