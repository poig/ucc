import torch
import torch.nn.utils.prune as prune
from ucc.noise_aware import CircuitFormer

# 1. Load your fully trained model
model_params = {
    "feature_dim": 16,
    "model_dim": 256,
    "n_heads": 8,
    "n_layers": 8,
    "dropout": 0.1,
    "max_seq_len": 1024,
}  # Your model parameters
trained_model = CircuitFormer(**model_params)
model_path = "C:/Users/junli/ucc/ucc/noise_aware/ml_model/trained_models_medium_reliable/best_model.pth"
trained_model.load_state_dict(torch.load(model_path))
trained_model.eval()

# 2. Define which layers to prune (typically linear layers in a Transformer)
parameters_to_prune = []
for name, module in trained_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, "weight"))

# 3. Apply pruning globally across all selected layers
#    Let's prune 30% of the weights with the smallest magnitude.
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,  # Prune 30% of connections
)

# 4. IMPORTANT: Make the pruning permanent and remove the pruning "mask"
#    This reduces the model's size on disk and can speed up inference.
for module, name in parameters_to_prune:
    prune.remove(module, name)

# 5. Save the new, smaller, pruned model
pruned_model_path = "C:/Users/junli/ucc/ucc/noise_aware/ml_model/trained_models_medium_reliable/pruned_model.pth"
torch.save(trained_model.state_dict(), pruned_model_path)
print("Model pruned and saved!")
