import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_size = embed_size
        self.heads = heads 
        
        assert embed_size % heads == 0, 'Embed size needs to be divisible by number of heads'

        self.head_dim = embed_size // heads

        # Note that in the following, head_dim * heads is equal to embed_size
        # However, this better expresses that each linear layer is technically self.heads linear layer concatenated
        # Each working on a subsection of an imput
        self.values = nn.Linear(
            in_features=self.head_dim * self.heads,
            out_features=self.head_dim * self.heads, 
            bias=False)
        self.keys = nn.Linear(
            in_features=self.head_dim * self.heads, 
            out_features=self.head_dim * self.heads, 
            bias=False)
        self.queries =nn.Linear(
            in_features=self.head_dim * self.heads, 
            out_features=self.head_dim * self.heads, 
            bias=False)

        self.fc_out = nn.Linear(self.heads * self.head_dim, embed_size)

    def forward(self, 
                values: torch.Tensor, keys: torch.Tensor, queries: torch.Tensor, mask):
        batch_size = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # pass the K, V and Q through multi-head attention layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Split ebedding into self.heads pieces
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, query_len, self.heads, self.head_dim)

        # N is batch size
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy (attention strenght) shape: (N, heads, query_len, key_len) 
        # i.e. for each head, cross-importance between each word of the target (query_len) and source (key_len)
        # einsum is magic, you tell it how you want dimensions to behave, and it makes sense of that and just multiplies everything
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            try:
                energy = energy.masked_fill(mask == 0, float('-1e20'))  # this makes it be 0 after softmax layer; should be -inf but underflow problems
            except RuntimeError as e:
                print('oh')
                energy = energy

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)  # make them add-up to 1 by the last dim

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_heads_dim)
        # desired shape: (N, query_len, heads, heads_dim)
        out = torch.einsum('nhqi,nihd->nqhd', [attention, values]).reshape(
            batch_size, query_len, self.heads*self.head_dim  # aka self.embed_size
        )

        out = self.fc_out(out)
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_size, heads, 
                 dropout, forward_expansion, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))  # sum with query for skip conn

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # sum with x for skip conn

        return out


class Encoder(nn.Module):
    def __init__(self,
                 embed_size,
                 heads,
                 num_layers,
                 src_vocab_size,
                 max_length=512,
                 forward_expansion=4,
                 dropout=0.2,
                 device='cpu' ) -> None:
        
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embeding = nn.Embedding(max_length, embed_size)
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            ) for _ in range(self.num_layers) 
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, seq_len = x.shape
        positions = (
            torch.arange(0, seq_len)
            .expand(batch_size, seq_len)
            .to(self.device)
        )

        x = self.dropout(
            self.word_embedding(x) + self.position_embeding(positions)
        )

        for layer in self.layers:
            out = layer(x, x, x, mask)  # since we are in the encoder K, V and Q are all the same (input sequence)

        return out 
    

class DecoderBlock(nn.Module):
    def __init__(self, 
                 embed_size, heads, 
                 forward_expansion=4, 
                 dropout=0.2, device='cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.masked_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, 
                                                  dropout, forward_expansion)
        

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.masked_attention(x, x, x, target_mask)
        query = self.dropout(self.norm(
            attention + x
        ))
        out = self.transformer_block(value, key, query, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(self, 
                 embed_size,
                 heads,
                 num_layers,
                 forward_expansion,
                 target_vocab_size,
                 max_length, 
                 dropout=0.2,
                 device='cpu',
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embeding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, 
                         forward_expansion, dropout, device) 
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, target_mask):
        batch_size, seq_len = x.shape 
        positions = (
            torch.arange(0, seq_len)
            .expand(batch_size, seq_len)
            .to(self.device)
        )

        x = self.dropout(
            self.word_embedding(x) + self.position_embeding(positions)
        )

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)

        out = self.fc_out(x)

        return out
    
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size, target_vocab_size,
                 src_pad_idx, target_pad_idx,
                 embed_size=256,
                 heads=8,
                 forward_expansion=4,
                 num_layers=6,
                 dropout=0,
                 device='cpu',
                 max_length=100):
        
        super().__init__()

        self.encoder = Encoder(
            embed_size, 
            heads, 
            num_layers, 
            src_vocab_size, 
            max_length, 
            forward_expansion, 
            dropout, 
            device
        )

        self.decoder = Decoder(
            embed_size,
            heads,
            num_layers,
            forward_expansion,
            target_vocab_size,
            max_length, 
            dropout,
            device
        )

        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device 


    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = (
            torch.tril(
                torch.ones(target_len, target_len)
            ).expand(
                N, 1, target_len, target_len
            )
        )

        return target_mask.to(self.device)
    
    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)

        return out
    

if __name__ == '__main__':
    device = 'cpu'

    x = torch.randint(10, size=(2, 9)).to(device)
    target = torch.randint(10, size=(2, 8)).to(device)  # doesn't need to be same shape as input

    src_pad_idx = 0
    target_pad_idx = 0
    src_vocab_size = 10
    target_vocab_size = 10

    model = Transformer(src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx).to(device)

    out = model(x, target[:, :-1])  # fictionally, we remove the stop token

    print(out.shape)