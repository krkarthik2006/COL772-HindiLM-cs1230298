import math
import torch
import torch.nn as nn
from typing import Any, Dict, List
from torch.nn.utils.rnn import pad_sequence

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x_norm
    
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden_dim = int(8 * d_model / 3) 
        self.w12 = nn.Linear(d_model, 2 * hidden_dim, bias=False) 
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(nn.functional.silu(x1) * x2)
    
    
def precompute_rope_angles(dim: int, seq_len: int, device: torch.device, base: int = 10000):
    """Precomputes the cosine and sine matrices for Rotary Positional Embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb), torch.sin(emb)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to a tensor of shape (B, seq_len, n_heads, d_head)."""
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, d_head)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, d_head)
    
    d = x.shape[-1]
    x_rot = torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)
    return x * cos + x_rot * sin
class multi_head_attention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_head = self.d_model // self.n_heads
        
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model , self.d_model,bias=False)
        
        self.attn_dropout = nn.Dropout(config.get('attention_dropout', 0.1))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor,cos : torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        head_outputs = []
        B, seq_len, _ = x.size()
        #print(x.shape)
        query_matrix = self.W_q(x).view(B, seq_len, self.n_heads, self.d_head)
        #print(query_matrix.shape)
        key_matrix = self.W_k(x).view(B, seq_len, self.n_heads, self.d_head)
        value_matrix = self.W_v(x).view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        query_matrix = apply_rope(query_matrix, cos, sin).transpose(1, 2)
        key_matrix = apply_rope(key_matrix, cos, sin).transpose(1, 2)
        S = torch.matmul(query_matrix , key_matrix.transpose(-2, -1))/ math.sqrt(self.d_head)
        if (self.config['mode']== "tanh-clipped"):
            S= self.config["tau"]*torch.tanh(S)
        
        #m wiht upper triangluar matrix as -inf rest 0 
        M = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        S = S+ M.unsqueeze(0).unsqueeze(0)
        
        #padding masking 
        pad_mask = (attention_mask == 0).view(B, 1, 1, seq_len)
        S = S.masked_fill(pad_mask, float('-inf'))
        
        Attention = nn.Softmax(dim=-1)(S)
        Attention = self.attn_dropout(Attention)
        head_outputs = torch.matmul(Attention, value_matrix)
        
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        output = self.W_o(head_outputs)
        return output
        

class transformer_block(nn.Module):
    def __init__(self, config: Dict[str, Any],layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_norm1 = RMSNorm(config['d_model'])
        self.layer_norm2 = RMSNorm(config['d_model'])
        self.feed_forward = SwiGLUFeedForward(config['d_model'])
        
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        self.multi_head_attention = multi_head_attention(config)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor,cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        org_x = x
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x, attention_mask, cos, sin)
        x= self.dropout(x)
        x = x + org_x
        org_x = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + org_x
        return x
        

class LanguageModel(nn.Module):
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        super().__init__()
        d_model = config['d_model']
        self.input_embeddings = nn.Embedding(num_embeddings=self.config['vocab_size'], embedding_dim=self.config['d_model'])
        self.embedding_dropout = nn.Dropout(config.get('dropout', 0.1))
        self.transformer_blocks = nn.ModuleList([transformer_block(config, i) for i in range(config['n_layers'])])
        self.layer_norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, config['vocab_size'], bias=False)
        
        if not config.get('no_tie_embeddings', False):
            self.lm_head.weight = self.input_embeddings.weight
        self.probabilites = None
        self.apply(self._init_weights)
        

            
    def positional_encoding(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        d_model = self.config['d_model']
        device = input_ids.device

        pos = torch.arange(seq_len,device = device).unsqueeze(1)
        i = torch.arange(0, d_model, 2, device=device)

        div_term = 10000 ** (i / d_model)

        angles = pos / div_term

        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return pe
            

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        with torch.no_grad():
            self.input_embeddings.weight.copy_(weights['W_vocab'].T)
            if not getattr(self.lm_head.weight, "data_ptr") == getattr(self.input_embeddings.weight, "data_ptr"):
                self.lm_head.weight.copy_(weights["W_devocab"].T)
            
            self.layer_norm_final.weight.copy_(weights['gamma_final'])
            # self.layer_norm_final.bias.copy_(weights['beta_final'])
        
        for i,block in enumerate(self.transformer_blocks):
            layer = i+1
            with torch.no_grad():
                block.layer_norm1.weight.copy_(weights[f'gamma_{layer}_1'])
                # block.layer_norm1.bias.copy_(weights[f'beta_{layer}_1'])
                block.layer_norm2.weight.copy_(weights[f'gamma_{layer}_2'])
                # block.layer_norm2.bias.copy_(weights[f'beta_{layer}_2'])

                # block.up_proj.weight.copy_(weights[f'W_{layer}_up'].T)
                
                # block.up_proj.bias.copy_(weights[f'b_{layer}_up'])
                # block.down_proj.weight.copy_(weights[f'W_{layer}_down'].T)
                # block.down_proj.bias.copy_(weights[f'b_{layer}_down'])

            W_q_list = []
            W_v_list = []
            W_k_list = []
            for head in range(self.config['n_heads']):
                head_idx = head+1
                W_q_list.append(weights[f'W_{layer}_Q_{head_idx}'])
                W_k_list.append(weights[f'W_{layer}_K_{head_idx}'])
                W_v_list.append(weights[f'W_{layer}_V_{head_idx}'])
            with torch.no_grad():
                block.multi_head_attention.W_q.weight.copy_(torch.cat(W_q_list, dim=0).T)
                block.multi_head_attention.W_k.weight.copy_(torch.cat(W_k_list, dim=0).T)
                block.multi_head_attention.W_v.weight.copy_(torch.cat(W_v_list, dim=0).T)

                block.multi_head_attention.W_o.weight.copy_(weights[f'W_{layer}_O'].T)
        


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """
        seq_len = input_ids.size(1)
        #1 input encoding- in init
        #2 positional encoding
        encodings = self.input_embeddings(input_ids) 
        encodings = self.embedding_dropout(encodings)
        cos, sin = precompute_rope_angles(self.config['d_model'] // self.config['n_heads'], seq_len, input_ids.device)
        #3 transformer blocks
        for block in self.transformer_blocks:
            encodings = block(encodings, attention_mask, cos, sin)
        #4 final norm
        encodings = self.layer_norm_final(encodings)
        #5 projection to vocab size
        logits = self.lm_head(encodings)
        #6 softmax
        # self.probabilites = nn.Softmax(dim=-1)(logits)
        
        #unnorm prob ret
        return logits


def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """
    
    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  
    padded_input_ids = pad_sequence(batch['input_ids'], batch_first=True, padding_value=PAD_ID).long()
    padded_attention_mask = pad_sequence(batch['attention_mask'], batch_first=True, padding_value=PAD_ID).long()
    return {'input_ids': padded_input_ids, 'attention_mask': padded_attention_mask}
