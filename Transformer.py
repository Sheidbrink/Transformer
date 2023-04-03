import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass

class Transformer(nn.Module):
    def __init__(self, dim_vocab, dim_pos, dim_embed, dim_ff, n_heads, n_enc_layers, n_dec_layers, activation=nn.GELU, do_pre_norm=True):
        '''
        Paramers
        --------
        dim_vocab: int
            Size of the vocabulary
        dim_pos: int
            max length, or at least anticipated one as it depends ultimately on the embedding
        dim_embed: int
            dimension of the embeddings
        dim_ff: int
            dimension of the hidden layer of the feed forward network
        n_heads: int
            number of attention heads
        n_enc_layers: int
            number of encoder layers
        n_dec_layers: int
            number of decoder layers
        activation: Callable nn.Module
            used to create the activation layer
        do_pre_norm: bool
            Whether to do the normilization before the layer (True) or after (False)
        '''
        super().__init__()
        self.dim_pos = dim_pos
        self.tok_embed = nn.Embedding(dim_vocab, dim_embed)
        # TODO Make positional embedding swapable
        self.pos_embed = nn.Embedding(dim_pos, dim_embed)
        
        # partial funcs just to make the `self.encoder =` line clean
        EncoderBlock = partial(TransformerBlock,
                               dim_embed = dim_embed,
                               n_heads = n_heads,
                               dim_ff = dim_ff,
                               use_self_causal_mask = False,
                               add_cross_attn = False,
                               activation = activation,
                               do_pre_norm = do_pre_norm)
        DecoderBlock = partial(TransformerBlock,
                               dim_embed = dim_embed,
                               n_heads = n_heads,
                               dim_ff = dim_ff,
                               use_self_causal_mask = True,
                               # If this is just a decoder then you don't need the other attn
                               add_cross_attn = n_enc_layers > 0,
                               activation = activation,
                               do_pre_norm = do_pre_norm)
        self.n_enc_layers, self.n_dec_layers = n_enc_layers, n_dec_layers
        self.encoder = nn.Sequential(*[EncoderBlock() for _ in range(n_enc_layers)])
        self.decoder = nn.ModuleList([DecoderBlock() for _ in range(n_dec_layers)])
        # TODO do projection out with Embedding weights
        self.lm_head = nn.Linear(dim_embed, dim_vocab)
        self.lm_head.weight = self.tok_embed.weight # I don't think this works in 2.0

    def forward(self, x, targets=None):
        n_batch, n_seq = x.shape
        tok_emb = self.tok_embed(x)
        pos_emb = self.pos_embed(torch.arange(n_seq, device=x.device))
        x = tok_emb + pos_emb
        
        # Only one of these paths is ever taken based on how the Transformer was init'd
        # TODO remove these checks because ^...somehow
        if self.n_enc_layers > 0:
            x_e = self.encoder(x)
            if self.n_dec_layers > 0:
                tok_emb = self.tok_embed(targets)
                pos_emb = self.pos_embed(torch.arange(targets.shape[1], device=x.device))
                tgts = tok_emb + pos_emb
                x = self.decode(tgts, x_e)
            else:
                x = x_e
        else:
            x = self.decode(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T,C), targets.view(B*T))

        return logits, loss
    
    def decode(self, x, mem=None):
        for block in self.decoder:
            if mem is None:
                x = block(x)
            else:
                x = block(x, mem)
        return x

    def generate(self, idx, max_new_tokens):
        # TODO temperature, beam search
        for _ in range(max_new_tokens):
            idx_ctx = idx[:, -self.dim_pos:] # [:, -block_size:] only feed in context size
            logits, loss = self(idx_ctx)
            logits = logits[:,-1,:]
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim_embed, n_heads, dim_ff,
                 use_self_causal_mask=True,
                 add_cross_attn=False,
                 activation=nn.GELU,
                 do_pre_norm=False):
        super().__init__()
        self.do_pre_norm = do_pre_norm
        # TODO make attention mechanism swapable? RWKV?
        self.self_attn  = MultiHeadAttention(dim_embed, n_heads,
                                             is_causal=use_self_causal_mask)
        self.ff         = MLP(dim_embed, dim_ff=dim_ff,
                              activation=activation,
                              n_layers=1)
        self.self_norm  = nn.LayerNorm(dim_embed)
        self.ff_norm    = nn.LayerNorm(dim_embed)
        if add_cross_attn:
            self.cross_attn = MultiHeadAttention(dim_embed, n_heads)
            self.cross_norm = nn.LayerNorm(dim_embed)
            self.forward = self.decoder_forward
        else:
            self.forward = self.encoder_forward

    def decoder_forward(self, x, enc):
        if self.do_pre_norm:
            x = x + self.self_attn(self.self_norm(x))
            x = x + self.cross_attn(self.cross_norm(x), enc)
            x = x + self.ff(self.ff_norm(x))
        else:
            x = x + self.self_norm(self.sa(x))#, src_mask))
            x = x + self.cross_norm(self.cross_attn(x, enc))
            x = x + self.ff_norm(self.ff(x))
        return x

    def encoder_forward(self, x):
        if self.do_pre_norm:
            x = x + self.self_attn(self.self_norm(x))
            x = x + self.ff(self.ff_norm(x))
        else:
            x = x + self.self_norm(self.sa(x))
            x = x + self.ff_norm(self.ff(x))
        return x

class MLP(nn.Module):
    def __init__(self, dim_embed, dim_ff=None, activation=nn.GELU, n_layers=1):
        super().__init__()
        if dim_ff is None:
            dim_ff = 4 * dim_embed
        self.net = nn.Sequential()

        dim_out = dim_embed
        for cur_layer in range(n_layers):
            is_last = cur_layer == n_layers-1

            dim_in  = dim_out
            dim_out = dim_embed if is_last else dim_ff

            self.net.append(nn.Linear(dim_in, dim_ff))
            self.net.append(activation())
            self.net.append(nn.Linear(dim_ff, dim_out))

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embed, n_heads, dim_heads=None, is_causal=False):
        super().__init__()
        if dim_heads is None:
            dim_heads = dim_embed
        assert dim_heads % n_heads == 0
        self.n_heads = n_heads
        
        self.query  = nn.Linear(dim_embed, dim_heads)
        self.key    = nn.Linear(dim_embed, dim_heads)
        self.value  = nn.Linear(dim_embed, dim_heads)
        self.norm = dim_heads ** -.5
        self.proj = nn.Linear(dim_heads, dim_embed)
        self.is_causal = is_causal
        
        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k

        # Split heads so that attention of different parts can be on different words
        #     batch x nheads x seq_len x head_dim
        q = self.split_heads(self.query(q))
        k = self.split_heads(self.key(k))
        v = self.split_heads(self.value(v))
        
        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v,
                                                 attn_mask=None, dropout_p=0.0,
                                                 is_causal=self.is_causal)
        else:
            alp = q @ k.transpose(-2, -1) * self.norm
            SEQ_LEN = alp.shape[2]
            if self.is_causal:
                mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN)))
            if mask is not None:
                mask = mask.view(1, 1, SEQ_LEN, SEQ_LEN)
                alp = alp.masked_fill(mask == 0, float('-inf'))
            alp = F.softmax(alp, dim=-1)
            out = alp @ v
        out = self.join_heads(out)
        return self.proj(out)

    def split_heads(self, x):
        n_batch, n_seq, dim_heads = x.shape
        x = x.view(n_batch, n_seq, self.n_heads, dim_heads // self.n_heads)
        return x.transpose(1, 2)

    def join_heads(self, x):
        n_batch, n_heads, n_seq, dim_head = x.shape
        return x.transpose(1,2).contiguous().view(n_batch, n_seq, n_heads * dim_head)