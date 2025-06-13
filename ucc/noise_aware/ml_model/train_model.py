# File: run_training_standalone.py

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
import math
from qiskit.dagcircuit import DAGCircuit
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from ..noise_aware_pass import DeviceNoiseProfile
from ..backend_utils import get_target
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from torch_geometric.loader import DataLoader as GeometricDataLoader

# ==============================================================================
# 1. MODEL ARCHITECTURE (Copied directly into this file)
# ==============================================================================


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [batch_size, seq_len, embedding_dim]"""
        # The slice of self.pe must match the sequence length of x
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CircuitFormer(nn.Module):
    """
    An Encoder-Only Transformer model to predict quantum circuit fidelity.
    It processes a sequence of rich gate feature vectors.
    """

    def __init__(
        self,
        feature_dim: int,
        model_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_seq_len: int,
    ):
        """
        Args:
            feature_dim (int): The dimensionality of the input feature vector for each gate.
            model_dim (int): The internal dimensionality of the Transformer (d_model).
            n_heads (int): The number of attention heads.
            n_layers (int): The number of stacked Transformer encoder layers.
            dropout (float): The dropout rate.
            max_seq_len (int): The maximum sequence length the model can handle.
        """
        super().__init__()
        # print(f"--- Model Instantiation ---")
        # print(f"DEBUG: CircuitFormer class received max_seq_len = {max_seq_len}")

        self.model_dim = model_dim
        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # This is where the positional encoding table size is determined.
        # It must be based on the max_seq_len of the *data* + 1 for the CLS token.
        self.pos_encoder = PositionalEncoding(
            model_dim, dropout, max_len=max_seq_len + 1
        )
        # print(f"DEBUG: PositionalEncoding created with max_len = {max_seq_len + 1}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Args: src: Tensor, shape [batch_size, seq_len, feature_dim]"""
        src = self.input_projection(src)
        cls_tokens = self.cls_token.expand(src.shape[0], -1, -1)
        x = torch.cat(
            (cls_tokens, src), dim=1
        )  # Shape becomes [batch_size, seq_len+1, model_dim]
        x = self.pos_encoder(x)
        encoded_output = self.transformer_encoder(x)
        cls_output = encoded_output[:, 0]
        prediction = self.classifier_head(cls_output)
        return prediction


class CircuitGNN(nn.Module):
    """
    A Graph Neural Network (GraphSAGE) model to predict quantum circuit fidelity.
    It operates directly on the graph representation of the circuit.
    """

    def __init__(
        self,
        feature_dim: int,  # Dimensionality of each node's features
        model_dim: int,  # The hidden dimension of the GNN layers
        n_layers: int,  # The number of GNN layers
        dropout: float,
    ):
        """
        Args:
            feature_dim (int): The number of features for each node (gate) in the graph.
            model_dim (int): The hidden dimensionality of the GraphSAGE layers.
            n_layers (int): The number of stacked GraphSAGE layers.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Input layer: maps raw features to the model's hidden dimension
        self.convs.append(SAGEConv(feature_dim, model_dim))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(model_dim, model_dim))

        # Classifier Head: takes the graph-level embedding and predicts a single value
        self.classifier_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: A PyTorch Geometric `Data` object containing:
                  - x: Node features, shape [num_nodes, feature_dim]
                  - edge_index: Graph connectivity, shape [2, num_edges]
                  - batch: Batch vector, shape [num_nodes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:  # No dropout on the last layer
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Pooling: Aggregate node features into a single graph-level feature vector
        # We use mean pooling here.
        graph_embedding = global_mean_pool(x, batch)

        # Pass the graph embedding through the final classifier
        prediction = self.classifier_head(graph_embedding)

        return prediction.view(-1)  # Flatten the output


def physical_dag_to_graph(
    physical_dag: DAGCircuit,
    noise_profile,
    gate_vocab: list,
    num_params: int,
    num_qubit_features: int,
    num_gate_cal_features: int,
) -> Data:
    """
    Converts a physically-mapped Qiskit DAGCircuit into a PyTorch Geometric
    Data object for use with a GNN.

    Nodes represent gates, and edges represent dependencies (wires).
    """
    nodes = list(physical_dag.op_nodes())
    node_map = {node: i for i, node in enumerate(nodes)}

    node_features = []
    edge_list = []

    for i, node in enumerate(nodes):
        # --- 1. Node Feature Extraction (same logic as before) ---
        # (This is a simplified version of your previous feature extractor)
        gate_type_encoding = [0.0] * len(gate_vocab)
        op_name = node.op.name
        if op_name in gate_vocab:
            gate_type_encoding[gate_vocab.index(op_name)] = 1.0
        else:
            gate_type_encoding[-1] = 1.0

        gate_params = [0.0] * num_params
        # ... (add your parameter extraction logic) ...

        [physical_dag.find_bit(q).index for q in node.qargs]
        phys_q1_features, phys_q2_features, gate_cal_features = (
            [0.0] * num_qubit_features,
            [0.0] * num_qubit_features,
            [0.0] * num_gate_cal_features,
        )
        # ... (add your noise profile feature extraction logic) ...

        feature_vector = (
            gate_type_encoding
            + gate_params
            + phys_q1_features
            + phys_q2_features
            + gate_cal_features
        )
        node_features.append(feature_vector)

        # --- 2. Edge Index Creation ---
        # Add edges from this node's predecessors to this node
        for pred in physical_dag.predecessors(node):
            if pred.type == "op":  # Only connect operation nodes
                pred_idx = node_map[pred]
                edge_list.append([pred_idx, i])

    # Convert to PyTorch Tensors
    x = torch.tensor(node_features, dtype=torch.float32)

    # edge_index must be shape [2, num_edges] and LongTensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Handle case with no edges (e.g., a circuit with one gate)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ==============================================================================
# 2. DATASET CLASS (Copied directly into this file)
# ==============================================================================


class FidelityDataset(Dataset):
    """
    A flexible PyTorch Dataset that can generate data for either a Transformer
    or a Graph Neural Network model.
    """

    def __init__(
        self,
        raw_data,  # A list of dictionaries: [{'dag': dag_obj, 'fidelity': score}, ...]
        noise_profile,  # The instantiated DeviceNoiseProfile object
        model_type: str,  # Either "transformer" or "gnn"
        # --- Transformer-specific arguments ---
        max_seq_len: int = 1024,
        # --- Common feature extraction arguments ---
        feature_dim: int = 16,
        gate_vocab: list = None,
        num_params: int = 1,
        num_qubit_features: int = 3,
        num_gate_cal_features: int = 2,
    ):
        if model_type not in ["transformer", "gnn"]:
            raise ValueError(
                "model_type must be either 'transformer' or 'gnn'"
            )

        self.raw_data = raw_data
        self.noise_profile = noise_profile
        self.model_type = model_type

        # Store all configuration parameters
        self.max_len = max_seq_len
        self.feature_dim = feature_dim
        self.gate_vocab = gate_vocab or [
            "cx",
            "sx",
            "rz",
            "x",
            "id",
            "measure",
            "other",
        ]
        self.num_params = num_params
        self.num_qubit_features = num_qubit_features
        self.num_gate_cal_features = num_gate_cal_features

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        # 1. Get the raw data point
        item = self.raw_data[idx]
        dag = item["dag"]  # The Qiskit DAGCircuit object
        fidelity_label = torch.tensor(
            [item["fidelity_label"]], dtype=torch.float32
        )

        # 2. Decide which feature extractor to use based on model_type
        if self.model_type == "transformer":
            # --- Generate padded tensor for the Transformer ---

            # This logic is moved from your old feature extractor directly into the dataset
            # It converts the DAG into a list of feature vectors first.
            feature_sequence = self._dag_to_feature_sequence(dag)

            # Then, it performs padding and truncation.
            if len(feature_sequence) > self.max_len:
                feature_sequence = feature_sequence[: self.max_len]

            padding_needed = self.max_len - len(feature_sequence)
            if padding_needed > 0:
                zero_vector = [0.0] * self.feature_dim
                feature_sequence.extend([zero_vector] * padding_needed)

            # The final item is the feature tensor
            final_data = torch.tensor(feature_sequence, dtype=torch.float32)

        elif self.model_type == "gnn":
            # --- Generate a graph Data object for the GNN ---
            final_data = physical_dag_to_graph(
                dag,
                self.noise_profile,
                gate_vocab=self.gate_vocab,
                num_params=self.num_params,
                num_qubit_features=self.num_qubit_features,
                num_gate_cal_features=self.num_gate_cal_features,
            )
            # For GNNs, the label is usually stored as an attribute of the Data object
            final_data.y = fidelity_label

        return final_data, fidelity_label

    def _dag_to_feature_sequence(self, physical_dag: DAGCircuit) -> list:
        """
        Helper method to generate the sequence of feature vectors for the Transformer.
        This contains the logic from your old _physical_dag_to_feature_tensor.
        """
        gate_feature_sequence = []
        for node in physical_dag.op_nodes():
            # (This is the exact same feature extraction logic as in physical_dag_to_graph)
            # --- Gate Type Encoding ---
            gate_type_encoding = [0.0] * len(self.gate_vocab)
            op_name = node.op.name
            if op_name in self.gate_vocab:
                gate_type_encoding[self.gate_vocab.index(op_name)] = 1.0
            else:
                gate_type_encoding[len(self.gate_vocab) - 1] = 1.0

            # --- Gate Parameter Extraction ---
            gate_params = [0.0] * self.num_params
            if hasattr(node.op, "params") and node.op.params:
                gate_params[0] = float(node.op.params[0]) / (2 * math.pi)

            # --- Noise Profile Feature Extraction ---
            physical_indices = [
                physical_dag.find_bit(q).index for q in node.qargs
            ]
            phys_q1_features = [0.0] * self.num_qubit_features
            phys_q2_features = [0.0] * self.num_qubit_features
            gate_cal_features = [0.0] * self.num_gate_cal_features
            if physical_indices:
                pq1 = physical_indices[0]
                t1, t2 = self.noise_profile.get_t1_t2(pq1)
                readout_err = self.noise_profile.get_readout_error(pq1)
                phys_q1_features = [t1, t2, readout_err]
                if len(physical_indices) == 2:
                    pq2 = physical_indices[1]
                    t1_2, t2_2 = self.noise_profile.get_t1_t2(pq2)
                    readout_err_2 = self.noise_profile.get_readout_error(pq2)
                    phys_q2_features = [t1_2, t2_2, readout_err_2]
                    gate_err, gate_dur = (
                        self.noise_profile.get_gate_properties(
                            op_name, (pq1, pq2)
                        )
                    )
                    gate_cal_features = [gate_err, gate_dur]
                elif len(physical_indices) == 1:
                    gate_err, gate_dur = (
                        self.noise_profile.get_gate_properties(op_name, (pq1,))
                    )
                    gate_cal_features = [gate_err, gate_dur]

            feature_vector = (
                gate_type_encoding
                + gate_params
                + phys_q1_features
                + phys_q2_features
                + gate_cal_features
            )
            gate_feature_sequence.append(feature_vector)

        return gate_feature_sequence


# ==============================================================================
# 3. MAIN TRAINING LOGIC (Copied from your script)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer for CircuitFormer with Early Stopping."
    )

    # --- Arguments for Data and Saving ---
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the JSON dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_models",
        help="Directory to save the trained model.",
    )

    # --- Arguments for Training Hyperparameters ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the AdamW optimizer.",
    )

    # --- Arguments for Early Stopping ---
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Epochs to wait for improvement before stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-5,
        help="Minimum change in validation loss to be considered an improvement.",
    )

    # --- Arguments for Model Architecture ---
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=16,
        help="Dimension of the gate feature vector.",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=128,
        help="Internal dimension of the Transformer.",
    )
    parser.add_argument(
        "--n-heads", type=int, default=8, help="Number of attention heads."
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="Number of Transformer encoder layers.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for model and data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        help="model transformer or gnn",
    )

    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # --- Load Data ---
    print(f"Loading dataset from: {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        raw_data = json.load(f)

    # 1. Select the model type from arguments
    model_type = "transformer" if args.model == "transformer" else "gnn"

    # 2. Create the appropriate dataset instance
    #    The FidelityDataset is now smart enough to handle both cases.
    print(f"Preparing data for '{model_type}' model...")
    target = FakeWashingtonV2()
    noise_profile = DeviceNoiseProfile(get_target(target))
    full_dataset = FidelityDataset(
        raw_data=raw_data,
        noise_profile=noise_profile,
        model_type=model_type,
        max_seq_len=args.max_seq_len,
        feature_dim=args.feature_dim,
        # You can pass other feature config args here if needed
    )

    # 3. Split the dataset into training and validation sets
    #    This part remains the same, as random_split works on any Dataset object.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    print(
        f"Loaded {len(raw_data)} samples. Training on {len(train_dataset)}, validating on {len(val_dataset)}."
    )

    # 4. Initialize the correct model AND the correct DataLoader
    if args.model == "transformer":
        print("Initializing CircuitFormer (Transformer) model...")
        model = CircuitFormer(
            feature_dim=args.feature_dim,
            model_dim=args.model_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=0.1,
            max_seq_len=args.max_seq_len,
        ).to(device)

        # Use the standard PyTorch DataLoader for the Transformer
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    elif (
        args.model == "gnn"
    ):  # Use "gnn", not "graphNN" to match your dataset's model_type
        print("Initializing CircuitGNN (Graph Neural Network) model...")
        model = CircuitGNN(
            feature_dim=args.feature_dim,
            model_dim=args.model_dim,
            n_layers=args.n_layers,
            dropout=0.1,
        ).to(device)

        # --- IMPORTANT: Use the DataLoader from PyTorch Geometric ---
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            # pin_memory is not typically used with PyG loaders
        )
        val_loader = GeometricDataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=4
        )

    else:
        raise ValueError(f"Unknown model type specified: {args.model}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )  # Mixed precision scaler

    print("\n--- Starting Model Training ---")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_log = []

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [Training]",
            leave=False,
        )

        for features, labels in train_pbar:
            features, labels = (
                features.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(features)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = (
                    features.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # --- Checkpointing & Early Stopping Logic ---
        if best_val_loss - avg_val_loss > args.min_delta:
            # Improvement found
            print(
                f"  -> Val loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}. Saving model..."
            )
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the new best model
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pth"),
            )
        else:
            # No improvement
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("\n--- Early Stopping Triggered ---")
            print(
                f"Validation loss has not improved for {args.patience} consecutive epochs."
            )
            break

    print("\n--- Training Complete ---")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")
    print(
        f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}"
    )
