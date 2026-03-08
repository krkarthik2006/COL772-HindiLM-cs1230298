import math
import torch
import torch.nn as nn
from typing import Any, Dict, List

class multi_head_attention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_head = self.d_model // self.n_heads
        
        self.W_q = nn.ModuleList([nn.Linear(self.d_model, self.d_head,bias = False) for _ in range(self.n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(self.d_model, self.d_head,bias = False) for _ in range(self.n_heads)])
        self.W_v = nn.ModuleList([nn.Linear(self.d_model, self.d_head,bias = False) for _ in range(self.n_heads)])
        self.W_o = nn.Linear(self.n_heads*self.d_model , self.d_model,bias = False)

    def mha_forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        head_outputs = []
        batch_size, seq_len, _ = x.size()
        for head_idx in range(self.n_heads):
            query_matrix = self.W_q[head_idx](x)
            key_matrix = self.W_k[head_idx](x)
            value_matrix = self.W_v[head_idx](x)

            S = query_matrix @ key_matrix.transpose(-2, -1) / math.sqrt(self.d_head)
            if (self.config['mode']== "tanh-clipped"):
                S= self.config["tau"]*torch.tanh(S)
            
            #m wiht upper triangluar matrix as -inf rest 0 
            M = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            S = S+M
            
            #padding masking 
            pad_mask = attention_mask.unsqueeze(1) == 0
            S = S.masked_fill(pad_mask, float('-inf'))
            
            Attention = nn.Softmax(S, dim=-1)
            head = Attention @ value_matrix
            head_outputs.append(head)
        
        final_concat_output = torch.cat(head_outputs, dim=-1)
        output = self.W_o(final_concat_output)
        return output
        

class transformer_block(nn.Module):
    def __init__(self, config: Dict[str, Any],layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config['d_model'],elementwise_affine=True)
        self.layer_norm2 = nn.LayerNorm(config['d_model'],elementwise_affine=True)
        self.up_proj = nn.Linear(config['d_model'], 4*config['d_model'])
        self.down_proj = nn.Linear(4*config['d_model'], config['d_model'])
        self.gelu = nn.GELU()
        self.multi_head_attention = multi_head_attention(config)
        

class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        super().__init__()
        d_model = config['d_model']
        self.input_embeddings = nn.Embedding(num_embeddings=self.config['vocab_size'], embedding_dim=self.config['d_model'])
        self.transformer_blocks = nn.ModuleList([transformer_block(config, i) for i in range(config['n_layers'])])
        self.lm_head = nn.Linear(d_model, config['vocab_size'], bias=False)
        self.layer_norm_final = nn.LayerNorm(d_model, elementwise_affine=True)
        self.probabilites = None

    def positional_encoding(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        d_model = self.config['d_model']

        pos = torch.arange(seq_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)

        div_term = 10000 ** (2 * i / d_model)

        angles = pos / div_term

        pe = torch.zeros(seq_len, d_model)
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
        self.input_embeddings.weight.data = weights['W_Vocab']
        self.lm_head.weight.data = weights["W_devocab"].T
        
        self.layer_norm_final.weight.data = weights['gamma_final']
        self.layer_norm_final.bias.data = weights['beta_final']
        
        for i,block in enumerate(self.transformer_blocks):
            layer = i+1
            block.layer_norm1.weight.data = weights[f'gamma_{layer}_1']
            block.layer_norm1.bias.data = weights[f'beta_{layer}_1']
            block.layer_norm2.weight.data = weights[f'gamma_{layer}_2']
            block.layer_norm2.bias.data = weights[f'beta_{layer}_2']
            
            block.up_proj.weight.data = weights[f'W_{layer}_up'].T
            block.up_proj.bias.data = weights[f'b_{layer}_up']
            block.down_proj.weight.data = weights[f'W_{layer}_down'].T
            block.down_proj.bias.data = weights[f'b_{layer}_down']
            for head in range(self.config['n_heads']):
                block.multi_head_attention.W_q.weight.data = weights[f'W_{layer}_Q_{head}']
                block.multi_head_attention.W_v.weight.data = weights[f'W_{layer}_V_{head}']
                block.multi_head_attention.W_k.weight.data = weights[f'W_{layer}_K_{head}']
            block.multi_head_attention.W_o.weight.data = weights[f'W_{layer}_O']
        
            
    def transformer_block(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        org_x = x
        x = self.layer_norm1(x)
        for 
        x = self.multi_head_attention.mha_forward(x, attention_mask)
        x = x + org_x
        org_x = x
        x = self.layer_norm2(x)
        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        x = x + org_x
        return x

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
        #1 input encoding- in init
        #2 positional encoding
        encodings = self.input_embeddings(input_ids) + self.positional_encoding(input_ids)
        #3 transformer blocks
        for _ in range(self.config['num_layers']):
            encodings = self.transformer_block(encodings, attention_mask)
        #4 final norm
        encodings = self.layer_norm_final(encodings)
        #5 projection to vocab size
        logits = encodings @ self.lm_head.weight.T
        #6 softmax
        self.probabilites = nn.Softmax(logits,axis=-1)
        
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


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID
    raise NotImplementedError("Implement collate_fn as described in assignment document")
