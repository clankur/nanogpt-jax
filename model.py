import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn

from typing import Optional, List, Tuple, Callable
KVCacheType = Optional[Tuple[jnp.ndarray, jnp.ndarray]]

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# making a mapping from character to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

@eqx.filter_value_and_grad
def cross_entropy (y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(y_true * jnp.log(y_pred))

class MultiHeadAttention(eqx.Module):
    num_heads: int
    head_size: int
    key: nn.Linear
    query: nn.Linear
    value: nn.Linear
    proj: nn.Linear
    dropout: nn.Dropout
    tril: jnp.ndarray

    def __init__ (self, jr_key: jnp.ndarray, num_heads: int, head_size: int) -> None:
        super().__init__()

        k_key, q_key, v_key, proj_key = jr.split(jr_key, 4)

        self.num_heads = num_heads
        self.head_size = head_size

        self.key = nn.Linear(n_embd, n_embd, use_bias=False, key=k_key)
        self.query = nn.Linear(n_embd, n_embd, use_bias=False, key=q_key)
        self.value = nn.Linear(n_embd, n_embd, use_bias=False, key=v_key)
        self.proj = nn.Linear(n_embd, n_embd, key=proj_key)
        self.dropout = nn.Dropout(dropout)

        self.tril = jnp.tril(jnp.ones((1, 1, block_size, block_size)))

    def forward (self, x: jnp.ndarray, use_cache: bool, kvcache: KVCacheType) -> Tuple[jnp.ndarray, KVCacheType]:
        B, T, C = x.shape
        k, v, q = self.key(x), self.value(x), self.query(x)  # [B, T, C]
        k, v, q = [t.reshape(B, T, self.num_heads, self.head_size).transpose(1, 2) for t in (k, v, q)]  # [B, T, C] -> [B, n, T, h]

        if use_cache:
            if kvcache:
                prev_k, prev_v = kvcache
                # truncate cache 
                prev_k, prev_v = prev_k[:, :, -block_size -1:, :], prev_v[:, :, -block_size-1:, :]
                k, v = jnp.concatenate([prev_k, k], axis=2), jnp.concatenate([prev_v, v], axis=2)
            kvcache = (k, v)
        
        att_wei = jnp.einsum('bnqh,bnkh->bnqk', q, k) * (int(self.head_size) ** -0.5)

        # casual mask
        att_wei = jnp.where(self.tril == 0, float('-inf'), att_wei) # sus
        att_wei = jax.nn.softmax(att_wei, axis=-1)
        att_wei = self.dropout(att_wei)

         # [B, n, Q, K] @ [B, n, K, h] -> [B, n, Q, h]
        out = jnp.einsum('bnqk,bnkh->bnqh', att_wei, v)
        out = out.transpose(1, 2).reshape(B, T, C) # [B, n, T, h] -> [B, T, C]
        out = self.proj(out) # communicate between different heads
        out = self.dropout(out)

        return out, kvcache

class FeedForward (eqx.Module):
    mlp: nn.Linear
    output: nn.Linear
    dropout: nn.Dropout

    def __init__ (self, key: jnp.ndarray, n_embd: int) -> None:
        super().__init__()
        linear_keys = jr.split(key, 2)

        # nn.Sequential is not working with jax.nn.relu for some reason
        # it gets 
        #   TypeError: zeros_like requires ndarray or scalar arguments,
        #   got <class 'jax._src.custom_derivatives.custom_jvp'>
        # so moved to the forward pass

        self.mlp = nn.Linear(n_embd, 4 * n_embd, key=linear_keys[0])
        self.output = nn.Linear(4 * n_embd, n_embd, key=linear_keys[1])
        self.dropout = nn.Dropout(dropout)

    def forward (self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = self.mlp(x)
        hidden = jax.nn.relu(hidden)

        output = self.output(hidden)
        output = self.dropout(output)

        return output

class Block (eqx.Module):
    heads: MultiHeadAttention
    ffwd: FeedForward
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__ (self, key: jnp.ndarray, n_embd: int, n_head: int) -> None:
        super().__init__()
        
        ffwd_key, att_key = jr.split(key, 2)
        
        head_size = n_embd // n_head
        self.heads = MultiHeadAttention(num_heads=n_head, head_size=head_size, jr_key=att_key)
        self.ffwd = FeedForward(key=ffwd_key, n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward (self, x: jnp.ndarray, use_cache: bool, kvcache:KVCacheType) -> Tuple[jnp.ndarray, KVCacheType]:
        heads_out, kvcache = self.heads(self.ln1(x), use_cache, kvcache)
        x = x + heads_out
        x = x + self.ffwd(self.ln2(x))
        return x, kvcache

class GPTLanguageModel (eqx.Module):
    block_size: int
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int

    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    blocks: list
    ln_f: nn.LayerNorm
    lm_head: nn.Linear

    def __init__ (self, key: jnp.ndarray, block_size: int, vocab_size: int, n_embd: int, n_head: int, n_layer: int) -> None:
        super().__init__()

        # split the key for seeding
        token_key, position_key, lm_key, blocks_key = jr.split(key, 4)
        blocks_key = jr.split(blocks_key, n_layer)

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, key=token_key)
        self.position_embedding_table = nn.Embedding(block_size, n_embd, key=position_key)
        self.blocks = [Block(n_embd=n_embd, n_head=n_head, key=blocks_key[i]) for i in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, key=lm_key)
    
    def __call__(self, idx: jnp.ndarray, targets: Optional[jnp.ndarray]=None, use_cache: bool=False, blocks_kvcache: List[KVCacheType]=[None] * n_layer) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], List[KVCacheType]]:
        return self.forward(idx, targets, use_cache, blocks_kvcache)

    def forward (self, idx: jnp.ndarray, targets: Optional[jnp.ndarray]=None, use_cache: bool=False, blocks_kvcache: List[KVCacheType]=[None] * n_layer) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], List[KVCacheType]]:
        print(idx.shape)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        history_length = 0 if not blocks_kvcache[0] else blocks_kvcache[0][0].shape[2]
        pos_emb = self.position_embedding_table(jnp.arange(T) + history_length)
        x = tok_emb + pos_emb
        new_kvcaches = []
        for block, kvcache in zip(self.blocks, blocks_kvcache):
            x, kvcache = block(x, use_cache, kvcache)
            new_kvcaches.append(kvcache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if not targets:
            loss = None
        else:
            B, T, C = logits.shape
            logits, targets = logits.reshape(B*T, C), targets.reshape(B*T)
            loss = cross_entropy(targets, logits)
        
        if use_cache:
            return logits, loss, new_kvcaches
        return logits, loss

    def generate (self, idx: jnp.ndarray, max_new_tokens: int) -> jnp.ndarray:
        curr_idx = idx
        blocks_kvcache = [None] * n_layer

        for _ in range(max_new_tokens):
            logits, _, blocks_kvcache = self(curr_idx, use_cache=True, blocks_kvcache=blocks_kvcache)
            last_token_logits = logits[:, -1, :]
            probs = jax.nn.softmax(last_token_logits, axis=-1)
            # sample and get the new token
            idx_next = jax.random.categorical(self.key, probs, axis=-1)

            curr_idx = idx_next
            idx = jnp.concatenate([idx, idx_next], axis=-1)
        return idx