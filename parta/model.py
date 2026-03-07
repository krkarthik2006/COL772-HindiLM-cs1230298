import math

import torch
import torch.nn as nn
from typing import Any, Dict, List

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
    
    def positional_encoding(self, input_ids: torch.Tensor) -> torch.Tensor:
        pe = torch.zeros(input_ids.size(1), self.config['d_model'])
        for pos in range(input_ids.size(1)):
            for i in range(0, self.config['d_model'], 2):
                pe[pos, 2*i] = math.sin(pos / (10000 ** (2*i / self.config['d_model'])))
                if i + 1 < self.config['d_model']:
                    pe[pos, 2*i + 1] = math.cos(pos / (10000 ** (2*(i) / self.config['d_model'])))
        return pe

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        
        raise NotImplementedError("Implement set_weights as described in assignment document")
    
    def layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        
    def multi_head_attention(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        
    def transformer_block(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        org_x = x
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x, attention_mask)
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
        #1 input encoding
        input_embeddings = nn.Embedding(num_embeddings=self.config['vocab_size'], embedding_dim=self.config['d_model'])(input_ids)
        #2 positional encoding
        encodings = input_embeddings + self.positional_encoding(input_ids)
        #3 transformer blocks
        for _ in range(self.config['num_layers']):
            encodings = self.transformer_block(encodings, attention_mask)
        #4 final norm
        encodings = self.layer_norm(encodings)
        #5 projection to vocab size
        logits = nn.Linear(encodings)(self.config['d_model'], self.config['vocab_size'])
        #6 softmax
        prob = nn.Softmax(logits,axis=-1)
        return prob


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
