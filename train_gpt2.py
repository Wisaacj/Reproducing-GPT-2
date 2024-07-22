import os
import inspect
import time
import sys
import tiktoken
import math
import torch
import torch.nn as nn
import torch.distributed as dist

from dataclasses import dataclass
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, and value projections for all heads, but in a batch.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Regularisation
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Not really a 'bias', more of a mask, but following teh OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(
            config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # The input is a tensor of shape (batch_size, seq_len, n_embd)
        # The output is a tensor of shape (batch_size, seq_len, n_embd)
        # The attention mask is a tensor of shape (1, 1, seq_len, seq_len)
        # The attention mask is a lower triangular matrix of ones.
        # The attention mask is broadcasted to the batch size.
        # The attention mask is broadcasted to the number of heads

        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality (n_embd)
        # Calculate query, key, values for all heads in one batch and move head forward
        # to be the batch dim. I.e., nh is a "batch dimension" meaning that PyTorch treats
        # B & nh as batches and it applies all the operations on both the batches and
        # heads in parallel. Notice that this is not *just* attention, this is multi-head attention.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # nh is the "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g., in GPT-2 (124M), n_head=12, hs=64, so nh*hs=768 channels in the Transformer.
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # # Attention (materialises the large (T, T) attention matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # This autoregressive mask makes sure that the tokens only attend to tokens before them
        # # and never to tokens in the future.
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # # Recall that attention multipled by the values is basically a way of doing a weighted
        # # sum of the values (the attention scores sum to 1).
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # The MLP is a two-layer feedforward neural network with a GELU activation funciton.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Guassian Error Linear Unit (GELU) is a non-linear activation function that is a
        # smooth version of the ReLU. We are using the tanh approximation because the original
        # GPT-2 implementation used it. Back then, calculating the GELU was expensive.
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Attention is a pooling function, a weighted aggregration function, a reduce function.
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # Whereas, a MLP happens to every token individually. The MLP is a map function.
        self.mlp = MLP(config)

    def forward(self, x):
        # The forward pass of the block is the forward pass of the two sublayers,
        # each with a residual skip connection.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # Max sequence length
    # Number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size: int = 50257
    n_layer: int = 12  # Number of layers
    n_head: int = 12  # Number of heads
    n_embd: int = 768  # Embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Weights of the token embeddings. Recall that nn.Embedding is really a
            # wrapper around a Tensor of shape (vocab_size, n_embd).
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # Weights of the position embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # h for hidden. Each block is the grey section of the decoder diagram in
            # Attention is All You Need (Vaswani et al., 2017).
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Finaly layer normalisation
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Finally, the langauge model head projects from n_embd to vocab_size with no bias.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme: the weights of the token embeddings and the final linear
        # layer are tied. This means that the weights of the token embeddings are the same
        # as the weights of the final linear layer.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # 2 residual layers per "Block". We use this scaling factor to stop the
                # variance of the activations from exploding. This is a trick that OpenAI
                # used in their implementation.
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # The input to the foward pass is a sequence of token indices.
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"""Cannot forward sequence of length {
            T}, block size is only {self.config.block_size}"""
        # Forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long,
                           device=idx.device)  # Shape (T,)
        # Position embeddings of shape (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # Token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # Forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # F.cross_entropy cannot handle 3D inputs, so we flatten the logits to 2D
            loss = F.cross_entropy(
                # Reshaping logits into (B*T, vocab_size)
                logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Loads pretrained GPT-2 model weights from huggingface
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            # 124M params
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # Always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257
        # Always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024
        # Create a from-scratch initialised minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(
            '.attn.bias')]  # Discard this mask

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            ".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        tranposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                     'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Basically, the OpenAI checkpoints ue a "Conv1D" module, but we only want to use a vanilla module.
        # This means we have to tranpose these weights when we import them.
        assert len(sd_keys_hf) == len(sd_keys), f"""mismatched keys: {
            len(sd_keys_hf)} != {len(sd_keys)}"""
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in tranposed):
                # Special treatment for the Conv1D weights we need to tranpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Start with all the candidate parameters (that required grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameter that is 2D will be weight decayed, otherwise it won't.
        # I.e., all weight tensors in matmuls + embeddings will decay, all biases and layernorms won't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available.
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(
            0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite:

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # At initialisation, load tokens from disk and store them in memory.
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)

        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")

        # State
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # Inputs
        y = (buf[1:]).view(B, T)  # Targets

        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # If loading the next batch would be out of bounds, reset.
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y


# -------------------------------------------------------------------------------------
# Simple launch:
# python3 train_gpt2.py
# DDP launch for e.g., 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# Set up DDP (distributed data parallel)
# torchrun commmand sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a DDP run?

if ddp:
    # Use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl')
    # Unique identifier for a process
    ddp_rank = int(os.environ['RANK'])
    # Used in a multi-node setting. Rank of a GPU on a single node
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # Total processes running
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # This process will do logging, checkpointing, etc.
    master_process = ddp_rank == 0
else:
    # Vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Attempt to auto-detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
B = 16  # Micro batch size
T = 1024  # Sequence length
assert total_batch_size % (
    B * T * ddp_world_size) == 0, "batch size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size:,} tokens")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Get a data batch
train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size)

# Enable TF32 for faster training
torch.set_float32_matmul_precision('high')

# Using vocab_size=50304 is a code performance optimisation (4% improvement). GPT-2 has a vocabulary size of 50257.
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    # Wrap the model in a distributed data parallel module. This module synchronises the gradients between ranks.
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    """Learning rate schedule"""
    # 1) Linear warmup for warmup_steps
    if it < warmup_steps:
        # Using (it + 1) because a learning rate of zero is not useful.
        return max_lr * (it+1) / warmup_steps
    # 2) If it > max_steps, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# AdamW is a bug fix for Adam. It's the same as Adam, but with a different weight decay.
# optimizer = torch.optim.AdamW(
#     model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # Normalise the loss to account for the gradient accumulation. We have to scale the loss
        # because the gradients just add on each successive backward(). Addition of gradients
        # corresponds to a SUM in the objective, but instead of a SUM we want MEAN. Scale the loss
        # here so it comes out right.
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        if ddp:
            # .backward() is a collective operation that synchronises the gradients across all the ranks.
            # It also scales the gradients by the number of processes. We only want to synchronise on the
            # final micro-step of the gradient accumulation.
            model.require_backward_grad_sync = (
                micro_step == grad_accum_steps - 1)
        # .backward deposits gradients: it accumulates the gradients from the loss
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # Gradient clipping is a technique to prevent exploding gradients in very deep networks.
    # A very bad batch of data can cause the gradients to explode and the model to fail to train.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Determine and set the learning rate for this iteration.
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Calling .step() updates the parameters based on the gradients.
    optimizer.step()
    # Wait for the GPU to finish all work scheduled to run.
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000  # Time difference in milliseconds
    tokens_processed = train_loader.B * \
        train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    if master_process:
        print(f"""step {step:4d} | loss: {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {
          dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}""".replace("\n", ""))

if ddp:
    destroy_process_group()

sys.exit(0)

model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)

# Prefix tokens (which are subsequently fed into the model as a prompt)
num_return_sequences = 5
max_length = 30
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)  # x is idx

torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Generate! Right now x is (B, T) where B = 5, T = 8
while x.size(1) < max_length:
    # Forward the model to get the logits. Wrapping this block in a torch.no_grad() context
    # manager indicates to PyTorch that we'll not be calling .backward() on any of the code.
    # This means PyTorch doesn't need to cache any of the intermediate Tensors and, subsequently,
    # that the operations within don't build up the computational graph. This makes the
    # inference faster and reduces the memory consumption.
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # Take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # Get the probabilities
        prob = F.softmax(logits, dim=-1)
        # Do top-k sampling of 50 (huggingface pipeline default). This ensures that we never
        # sample very unlikely tokens, but rather always sample from the top-k most likely tokens.
        # topk_probs here becomes (5, 50), topk_indices becomes (5, 50)
        topk_probs, topk_indices = prob.topk(50, dim=-1)
        # Select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
        # Gather the corresponding token indices
        xcol = topk_indices.gather(dim=-1, index=ix)  # (B, 1)
        # Append to the sequence
        x = torch.cat((x, xcol), dim=1)

# Print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
