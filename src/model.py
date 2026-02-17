import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
from typing import Optional
from model_args import ModelArgs


def precompute_theta_pos_embedding(h_dim: int,seq_len: int,device: str,theta: float=1000.0):

    # Head Dim Should be even
    assert h_dim % 2 == 0

    # 1000^(-2(i-1)/h_dim) -> i=1,2,3 ... dim/2
    numerator=torch.arange(start=0,end=h_dim,step=2).float()

    theta = 1.0/theta**(numerator/h_dim).to(device)

    m=torch.arange(start=0,end=seq_len).to(device)

    output=torch.outer(m,theta).float()

    freq_complex=torch.polar(torch.ones_like(output),output)
    return freq_complex

def apply_rotary_embeddings(x: torch.Tensor,freq_complex: torch.Tensor, device: str):
    # x : (B , Seq_Len , H , H_dim)
    b , sq, h , h_dim=x.shape
    x=x.reshape(b,sq,h,-1,2)
    x_complex=torch.view_as_complex(x)

    # (Seq_Len,Head_Dim/2) -> (1,Seq_Len,1,Head_Dim/2)
    freq_complex=freq_complex.unsqueeze(0).unsqueeze(2)

    x_rotated = x_complex * freq_complex

    x_out=torch.view_as_real(x_rotated)

    x_out=x_out.reshape(b,sq,h,h_dim)

    return x_out.type_as(x).to(device)


class Attention(nn.Module):

    def __init__(self,args: ModelArgs):
        super().__init__()

        self.num_head=args.n_heads
        self.d_model=args.d_model
        
        assert self.d_model % self.num_head == 0 , "D_Model Should be divisible by num_head"

        self.d_k=self.d_model // self.num_head


        self.w_q=nn.Linear(self.d_model,self.d_model)
        self.w_k=nn.Linear(self.d_model,self.d_model)
        self.w_v=nn.Linear(self.d_model,self.d_model)
        self.w_o=nn.Linear(self.d_model,self.d_model)


    def forward(self,x: torch.Tensor,freq_complex: torch.Tensor, hard_mask: torch.Tensor = None, soft_mask: torch.Tensor = None):
        # x: (Batch , Seq_Len , d_model)
        batch , seq_len , d_model = x.shape
        # x: (Batch , Seq_Len , d_model)
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)

        # (Batch , Seq_Len , d_model) -> (Batch , Seq_Len , n_head , d_k)
        q=q.view(batch , seq_len , self.num_head , self.d_k)
        k=k.view(batch , seq_len , self.num_head , self.d_k)

        # Add Rotatory Position Embedding (needs (B, Seq_Len, H, H_dim) format)
        q=apply_rotary_embeddings(q,freq_complex=freq_complex,device=x.device)
        k=apply_rotary_embeddings(k,freq_complex=freq_complex,device=x.device)

        # (Batch , Seq_Len , n_head , d_k) -> (Batch , num_head , Seq_Len , d_k)
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.view(batch , seq_len , self.num_head , self.d_k).transpose(1,2)
        attention = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.d_k)

        # Hard mask (causal): masked_fill_ with -inf BEFORE softmax -> softmax(-inf) = 0
        if hard_mask is not None:
            attention=attention.masked_fill_(hard_mask == 0,float('-inf'))

        attention = torch.softmax(attention,dim=-1)

        # Soft mask (learned): multiply AFTER softmax -> differentiable modulation
        if soft_mask is not None:
            attention=attention * soft_mask
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6) # ReNormalization
        attention = torch.matmul(attention,v)

        # (Batch , num_head , Seq_Len , d_k) -> (Batch , Seq_Len , num_head , d_k) -> (Batch , Seq_Len , d_model)
        output = attention.transpose(1,2).contiguous().view(batch,seq_len,d_model)
        # (Batch , Seq_Len , d_model)
        return self.w_o(output)
    
class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.d_model
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.d_model, bias=False)
        self.w3 = nn.Linear(args.d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x

class RMSNorm(nn.Module):

    def __init__(self , args: ModelArgs,eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        # Gama Parameter
        self.weight = nn.Parameter(torch.ones(args.d_model))

    
    def forward(self , x: torch.Tensor) -> torch.Tensor:

        # x: (B , Seq_Len , Dim)

        rms=torch.sqrt(x.pow(2).mean(dim =-1,keepdim=True) + self.eps)

        normalized_values = x / rms

        return self.weight * normalized_values 

class PredictMask(nn.Module):
    
    def __init__(self,args: ModelArgs):
        super().__init__()
        self.selfattention=Attention(args)
        self.norm_1=RMSNorm(args)
        self.scale_sigmoid_facor=args.scale_sigmoid_factor

        # Projects d_model -> max_Seq_len to produce (B, Seq_Len, Seq_Len) mask
        self.mask_proj=nn.Linear(args.d_model, args.max_Seq_len)

    def forward(self , x: torch.Tensor , freq_complex: torch.Tensor) -> torch.Tensor:

        seq_len = x.shape[1]

        # Causal mask: (1, 1, Seq_Len, Seq_Len) — 1 = attend, 0 = block future tokens
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        x = x + self.selfattention(self.norm_1(x),freq_complex,hard_mask = causal_mask)

        # x: (B , Seq_Len , d_model) -> (B , Seq_Len , max_Seq_len)
        mask = self.mask_proj(x)
        # Trim to actual seq_len -> (B , Seq_Len , Seq_Len)
        mask = mask[:, :, :seq_len]
        # Sharpen and squash to [0, 1]
        mask = F.sigmoid(mask * self.scale_sigmoid_facor)
        # (B , Seq_Len , Seq_Len) -> (B , 1 , Seq_Len , Seq_Len) to broadcast across heads
        mask = mask.unsqueeze(1)
        return mask


class ProcessWithLearnedMask(nn.Module):

    def __init__(self,args: ModelArgs):
        super().__init__()

        self.predictmask=PredictMask(args)
        self.attention=Attention(args)
        self.feed_forward=FeedForward(args)
        self.norm_1=RMSNorm(args)
        self.norm_2=RMSNorm(args)
    
    def forward(self,x:torch.Tensor , freq_complex: torch.Tensor):
        # x : (Batch , Seq_Len , d_model)
        batch , seq_len , d_model = x.shape
        soft_mask=self.predictmask(x,freq_complex)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        x = x + self.attention(self.norm_1(x),freq_complex,hard_mask=causal_mask,soft_mask=soft_mask)
        x = x + self.feed_forward(self.norm_2(x))

        return x



class Transformer(nn.Module):

    def __init__(self,args: ModelArgs):
        super().__init__()

        self.n_layer=args.n_layers
        self.layers=nn.ModuleList([])

        for _ in range(self.n_layer):
            self.layers.append(
                ProcessWithLearnedMask(args)
            )
        
        self.embedding = nn.Embedding(args.vocab_size,args.d_model)

        self.out_norm=RMSNorm(args)
        self.out_Proj=nn.Linear(args.d_model,args.vocab_size,bias=False)

        # Weight tying: share embedding weights with output projection (saves ~12.9M params)
        self.out_Proj.weight = self.embedding.weight
        self.register_buffer(
            "freq_complex",
            precompute_theta_pos_embedding(args.d_model // args.n_heads,args.max_Seq_len,device='cpu')
        )
       

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        # X : (Batch , Seq_Len)

        #  (Batch , Seq_Len) -> (Batch , Seq_Len , d_model)
        x = self.embedding(x)

        # Slice RoPE to actual seq_len (important for inference with variable lengths)
        seq_len = x.shape[1]
        freq_complex = self.freq_complex[:seq_len]

        for layer in self.layers:
            x=layer(x,freq_complex)
        
        x = self.out_norm(x)
        output = self.out_Proj(x)
        return output