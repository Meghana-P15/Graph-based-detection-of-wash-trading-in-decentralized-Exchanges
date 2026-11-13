"""Real 2-layer GCN and GraphSAGE implemented in pure PyTorch (no PyG required)."""
import torch
from torch import nn
from torch.nn import functional as F


class GCNLayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__(); self.linear=nn.Linear(in_dim,out_dim,bias=False); self.bias=nn.Parameter(torch.zeros(out_dim))

    def forward(self,x,edges):
        n=len(x)
        loop=torch.arange(n,device=x.device)
        idx=torch.cat([edges,torch.stack([loop,loop])],1)
        source,dest=idx[0],idx[1]
        deg=torch.bincount(dest,minlength=n).float().clamp_min(1)
        norm=deg[source].rsqrt()*deg[dest].rsqrt()
        h=self.linear(x)
        msg=h[source]*norm.unsqueeze(-1)
        out=torch.zeros((n,h.shape[1]),dtype=h.dtype,device=h.device)
        out.index_add_(0,dest,msg)
        return out+self.bias


class SAGELayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__();self.linear=nn.Linear(in_dim*2,out_dim)

    def forward(self,x,edges):
        src,dst=edges
        neigh=torch.zeros_like(x)
        neigh.index_add_(0,dst,x[src])
        cnt=torch.bincount(dst,minlength=len(x)).float().to(x.device).clamp_min(1)
        avg=neigh/cnt[:,None]
        return self.linear(torch.cat([x,avg],dim=-1))


class EventGNN(nn.Module):
    def __init__(self,arch="sage",input_dim=12,hidden_dim=48,dropout=.15):
        super().__init__()
        if arch not in ("gcn","sage"):
            raise ValueError("arch must be gcn or sage")
        self.arch=arch
        Layer=GCNLayer if arch=="gcn" else SAGELayer
        self.g1=Layer(input_dim,hidden_dim)
        self.g2=Layer(hidden_dim,hidden_dim)
        self.head=nn.Linear(hidden_dim,1)
        self.dropout=dropout

    def embed_nodes(self,x,edge_index):
        z=F.relu(self.g1(x,edge_index))
        z=F.dropout(z,p=self.dropout,training=self.training)
        return F.relu(self.g2(z,edge_index))

    def forward(self,x,edge_index,rootidx):
        z=self.embed_nodes(x,edge_index)
        return self.head(z[rootidx]).squeeze(-1),z[rootidx]
