import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import torch.nn.functional as F

class MultiLayerBetaDistributionNN(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes=None, learning_rate=0.001):
        super(MultiLayerBetaDistributionNN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [1028, 512, 256, 128]  # Default hidden layer sizes
        layers = []
        previous_size = input_size
        
        # Adding hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size
        
        # Adding the final layer to predict alpha and beta
        layers.append(nn.Linear(previous_size, 2))  # Output layer with 2 units
        
        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X, y_alpha, y_beta = batch
        y = torch.stack([y_alpha, y_beta], dim=1)  # Stack alpha and beta into a single tensor
        outputs = self(X)
        
        # Compute KL divergence loss
        q = torch.distributions.Beta(outputs[:, 0], outputs[:, 1])
        p = torch.distributions.Beta(y[:, 0], y[:, 1])
        loss = torch.distributions.kl_divergence(q, p).mean()
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y_alpha, y_beta = batch
        y = torch.stack([y_alpha, y_beta], dim=1)
        outputs = self(X)
        
        # Compute KL divergence loss
        q = torch.distributions.Beta(outputs[:, 0], outputs[:, 1])
        p = torch.distributions.Beta(y[:, 0], y[:, 1])
        loss = torch.distributions.kl_divergence(q, p).mean()
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
class MultiLayerNN(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes=None, learning_rate=0.001, use_bce_loss=False):
        super(MultiLayerNN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [1028, 512, 256, 128]  # Default hidden layer sizes

        layers = []
        previous_size = input_size

        # Adding hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        # Adding the final layer
        layers.append(nn.Linear(previous_size, 1))  # Output layer with a single unit
        layers.append(nn.Sigmoid())  # Sigmoid activation for probabilities

        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss() if use_bce_loss else nn.MSELoss(reduction="none")
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.view(-1, 1)  # Reshape y to [batch_size, 1]
        outputs = self(X)
        loss = self.criterion(outputs, y).mean()
        self.log('train_loss', loss, prog_bar=True)  # Log the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.view(-1, 1)  # Reshape y to [batch_size, 1]
        outputs = self(X)
        loss = self.criterion(outputs, y).mean()
        self.log('val_loss', loss, prog_bar=True)  # Log the validation loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
class NextTokenValueEstimationNN(pl.LightningModule):
    def __init__(self, model_id: str, consistency_weight: float = 1.0, initialize_weights_with_lm_head: bool=False, device: str="cuda:0"):
        super().__init__()
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id)
        
        if "llama" not in model_id:
            raise Exception("vocab, hidden_size and lm head weight extraction only possible for llama models")
        

        self.lm_head_weight: torch.Tensor = model.embed_tokens.weight.clone().detach().to(device)
        self.lm_head_weight.requires_grad = False
        vocab_size, hidden_size = model.embed_tokens.weight.shape

        print(f"Using vocab_size: {vocab_size}, hidden_siz: {hidden_size} from {model_id}")
        linear_layer = nn.Linear(hidden_size, vocab_size)
        if initialize_weights_with_lm_head:
            with torch.no_grad():
                linear_layer.weight.copy_(self.lm_head_weight)
        self.model = nn.Sequential(
            linear_layer,
            nn.Sigmoid()
        )
        
        self.consistency_weight = consistency_weight
        
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def calculate_loss(self, batch, predictions):
        hidden_states, token_ids, token_values, expected_values, mask = batch
        # 1. Value Prediction Loss (using mask to handle variable lengths)
        batch_size = hidden_states.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, token_ids.size(1))
        
        predicted_values = predictions[batch_indices, token_ids]  # [batch_size, max_tokens]
        value_loss = F.mse_loss(predicted_values * mask, token_values * mask, reduction='sum')
        value_loss = value_loss / mask.sum()  # Normalize by actual number of values
        
        # 2. Consistency Loss
        with torch.no_grad():
            logits = torch.matmul(hidden_states, self.lm_head_weight.t())
            next_token_probs = F.softmax(logits, dim=-1)

        predicted_expected = torch.sum(predictions * next_token_probs, dim=-1)
        consistency_loss = F.mse_loss(predicted_expected, expected_values)
        predicted_expected = torch.sum(predictions * next_token_probs, dim=-1)

        
        loss = value_loss + self.consistency_weight * consistency_loss
        return loss, value_loss, consistency_loss
    
    def training_step(self, batch, batch_idx):
        hidden_states, token_ids, token_values, expected_values, mask = batch
        
        # Get predictions for all next tokens
        predictions = self(hidden_states)  # [batch_size, vocab_size]
        loss, value_loss, consistency_loss = self.calculate_loss(batch, predictions)
        self.log("train_loss", loss, prog_bar=True)
        self.log("value_loss", value_loss)
        self.log("consistency_loss", consistency_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        hidden_states, token_ids, token_values, expected_values, mask = batch
        
        # Get predictions for all next tokens
        predictions = self(hidden_states)  # [batch_size, vocab_size]
        loss, value_loss, consistency_loss = self.calculate_loss(batch, predictions)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_value_loss", value_loss)
        self.log("val_consistency_loss", consistency_loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

class NextTokenValueAdapterNN(pl.LightningModule):
    def __init__(self, model_id: str, initialize_weights_with_lm_head: bool=False, device: str="cuda:0"):
        super().__init__()
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id)
        
        if "llama" not in model_id:
            raise Exception("vocab, hidden_size and lm head weight extraction only possible for llama models")
        

        self.lm_head_weight: torch.Tensor = model.embed_tokens.weight.clone().detach().to(device)
        self.lm_head_weight.requires_grad = False
        vocab_size, hidden_size = model.embed_tokens.weight.shape

        print(f"Using vocab_size: {vocab_size}, hidden_siz: {hidden_size} from {model_id}")
        linear_layer = nn.Linear(hidden_size, vocab_size)
        if initialize_weights_with_lm_head:
            with torch.no_grad():
                linear_layer.weight.copy_(self.lm_head_weight)
                
        self.model = nn.Sequential(
            linear_layer
        )
            
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def calculate_loss(self, batch, predictions):
        hidden_states, token_ids, token_values, expected_values, mask = batch
        # 1. Value Prediction Loss (using mask to handle variable lengths)
        batch_size = hidden_states.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, token_ids.size(1))
        
        predicted_values = predictions[batch_indices, token_ids]  # [batch_size, max_tokens]
        value_loss = F.l1_loss(predicted_values * mask, token_values * mask, reduction='sum')
        value_loss = value_loss / mask.sum()  # Normalize by actual number of values
        return value_loss
    
    def training_step(self, batch, batch_idx):
        hidden_states, token_ids, token_values, expected_values, mask = batch
        
        # Get predictions for all next tokens
        predictions = self(hidden_states)  # [batch_size, vocab_size]
        loss = self.calculate_loss(batch, predictions)
        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        hidden_states, token_ids, token_values, expected_values, mask = batch
        
        # Get predictions for all next tokens
        predictions = self(hidden_states)  # [batch_size, vocab_size]
        loss = self.calculate_loss(batch, predictions)
        self.log("val_loss", loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
def collate_fn(batch: list[tuple[torch.Tensor, list[int], list[float], float]]):
    """
    Handles variable-length token lists through padding.
    
    Args:
        batch: list of tuples (hidden_state, token_ids, token_values, expected_value)
            - hidden_state: tensor[hidden_size]
            - token_ids: list[int] of variable length
            - token_values: list[float] of same length as token_ids
            - expected_value: float
    """
    # Find max length of token lists
    max_tokens = max(len(item[1]) for item in batch)
    
    # Pad token_ids and token_values
    def pad_sequence(seq: list, max_len: int, pad_val: float) -> torch.Tensor:
        return torch.tensor(seq + [pad_val] * (max_len - len(seq)))
    
    hidden_states = torch.stack([item[0] for item in batch])
    token_ids = torch.stack([
        pad_sequence(item[1], max_tokens, 0) for item in batch
    ])
    token_values = torch.stack([
        pad_sequence(item[2], max_tokens, 0.0) for item in batch
    ])
    expected_values = torch.tensor([item[3] for item in batch])
    
    # Create mask for valid tokens (1 for real values, 0 for padding)
    mask = torch.stack([
        torch.tensor([1.0] * len(item[1]) + [0.0] * (max_tokens - len(item[1])))
        for item in batch
    ])
    
    return hidden_states, token_ids, token_values, expected_values, mask
# Example usage:
"""
# Dataset should return tuples:
dataset = [
    (
        hidden_state,           # tensor[hidden_size]
        token_ids,             # tensor[N] - indices of tokens with known values
        token_values,          # tensor[N] - corresponding values
        expected_value         # float
    ),
    ...
]

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_fn
)
"""