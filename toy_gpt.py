import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1400)

#set processing sizes
# ex. for seq 1 2 3 4 5 6 7,
# when input context is 1, target = 2
# [1, 2] --> 3
# [1, 2, 3] --> 4
# etc. --> develop predictions/connections from single character through block-size
# w/ transformer (for the batch and block_sizes below --> 32*8 separate examples)
block_size = 256 # max context length for pred
batch_size = 64 #num indep. seqs that are processed in parallel
interval = 200
iters = 2000
eval_iters = 100
embed_size = 384
dropout = 0.2
head_num = 6


def get_batch(split_type):
    if split_type == "train":
        data = train_data
    else:
        data = val_data
        
    #generate random offsets/indices (n = batch_size) to pull from data - between 0 and [len(data) - block_size] (max position in data)
    seq_indices = torch.randint(len(data) - block_size, (batch_size,))
    
    #stack 1d tensors as rows --> matrix
    x = torch.stack([data[i:i+block_size] for i in seq_indices])
    y = torch.stack([data[i+1:i+(block_size+1)] for i in seq_indices])
    return x, y
    
#get avg loss over batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #toggle to 'turn off' certain layers (e.g. dropout) and set how normalization stats are processed
    for split_type in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for j in range(eval_iters):
            x_, y_ = get_batch(split_type)
            _, loss = model(x_, y_)
            losses[j] = loss.item()
        out[split_type] = losses.mean()
        
    model.train() #reset mode
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias = False) #token-contained information
        self.query = nn.Linear(embed_size, head_size, bias = False)  #information being "sought"
        self.value = nn.Linear(embed_size, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b, t, c = x.shape
        key = self.key(x)
        query = self.query(x)
        #generate attention
        weights = q @ k.transpose(-2, -1) * (C**-0.5) #scaled attention
        tri = torch.tril(torch.ones(t, t))
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1)
        weights = self.dropout(weights)
        #weighted aggregation
        v = self.value(x)
        output = weights @ v
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for h in range(head_num)])
        self.projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        output = torch.cat([h_(x) for h_ in self.heads], dim = -1)
        output = self.dropout(self.projection(output))
        return output
        
class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(embed_size, 4*embed_size), nn.ReLU(), nn.Linear(4*embed_size, embed_size), nn.Dropout(dropout),)
    def forward(self, x):
        return self.network(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_num):
        super().__init__()
        head_size = embed_size // head_num
        self.attention = MultiHeadAttention(head_num, head_size)
        self.ff = FeedForward(embed_size)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
    def forward(self, x):
        x = x + self.attention(self.layernorm1(x)) #residual connection
        x = x + self.ff(self.layernorm2(x)) #residual connection
        return x

class LM(nn.Module):
    def __init__(self, char_count):
        super().__init__()
        #the embedding is effectively a linear layer, where each row is a vector corresponding to the input index
        self.token_embedding_table = nn.Embedding(char_count, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(TransformerBlock(embed_size, head_num=4), TransformerBlock(embed_size, head_num=4), TransformerBlock(embed_size, head_num=4), nn.LayerNorm(embed_size))
        self.lm_head = nn.Linear(embed_size, char_count)
        self.attention_heads = MultiHeadAttention(4, embed_size//4)
        self.ff = FeedForward(embed_size)
        
    def forward(self, indices, targets=None):
        b, t = indices.shape
        #indices and targets = (batch, block (time)) tensor of integers
        embed_token = self.token_embedding_table(indices) #tensor of (batch, time, channels = vec_length (char_count)) --> needs to reshaped to B*T, C for loss computation
        embed_position = self.position_embedding_table(torch.arange(t))
        x = embed_token + embed_position
        x = self.blocks(x)
        logits = self.lm_head(embed_token) #tensor of b,t,char_count
        
        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets) #negative log-likelihood
        return logits, loss
        
    def generate(self, indices, max_gen_tokens):
        #indices = (b, t) (in context)
        for p in range(max_gen_tokens):
            #constraint from positional embedding
            indices_subset = indices[:, -block_size:]
            #forward pass, create preds
            logits, loss = self(indices_subset)
            logits = logits[:, -1, :] #retrieve last time step
            #gen probability distrib with softmax
            probabilities = F.softmax(logits, dim = -1)
            #sample from distribution
            next_index = torch.multinomial(probabilities, num_samples = 1)
            #add sample to seq
            indices = torch.cat((indices, next_index), dim = 1) #(b, t+1)
        return indices
        
#read in text
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    
#retrieve set of all unique chars
chars = sorted(list(set(text)))
char_count = len(chars) #THIS IS MORE GENERALLY A SIZE OF THE LEXICON, DEPENDING ON TOKEN USE (CHAR VS WORD VS SENTENCE)

#dicts: char to int
s_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_s = {i:ch for i, ch in enumerate(chars)}

#mapping functions
encode = lambda s: [s_to_int[char] for char in s]
decode = lambda ls_i: ''.join([int_to_s[i] for i in ls_i])

#store encoding in tensor
data = torch.tensor(encode(text), dtype = torch.long)
#simple split into train/val sets
split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

x_batch, y_batch = get_batch('train')
        
model_ = LM(char_count)
logits, loss = model_(x_batch, y_batch)
optimizer = torch.optim.AdamW(model_.parameters(), lr=1e-3)

for step in range(iters):
    if step % interval == 0:
        losses = estimate_loss()
        print('step - ' + str(step) + ', train loss - ' + str(losses['train']) + ', val loss - ' + str(losses['val']))

    x_batch, y_batch = get_batch('train')
    logits, loss = model_(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype = torch.long)
print(decode(model_.generate(context, max_gen_tokens=50)[0].tolist()))
