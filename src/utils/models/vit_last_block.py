import torch


# Extract the last encoder block and subsequent layers
class ViTLastBlock(torch.nn.Module):
    def __init__(self, original_model):
        super(ViTLastBlock, self).__init__()
        # Extract the last encoder block starting from the multihead attention
        self.multihead_attn = original_model.encoder.layers[-1].self_attention
        self.dropout = original_model.encoder.layers[-1].dropout
        self.mlp = original_model.encoder.layers[-1].mlp
        self.layernorm2 = original_model.encoder.layers[-1].ln_2

        self.head = original_model.heads.head

    def forward(self, x):
        # Pass through multihead attention
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x)
        x = x + attn_output
        x = self.dropout(x)
        x = self.layernorm2(x)
        # Pass through MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output

        x = self.head(x[:, 0])  # Only use the class token for prediction
        return x
