# File: run_training_standalone.py

import argparse
import json
import numpy as np
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
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from torch_geometric.loader import DataLoader as GeometricDataLoader
from neuralop.models import FNO

from qiskit.transpiler import Target
from qiskit.providers import Backend, BackendV2
from typing import Union


class DeviceNoiseProfile:
    """
    A container for a device's noise and topology characteristics.
    """

    def __init__(self, target: Target):
        if not isinstance(target, Target):
            raise TypeError("Input must be a Qiskit Target instance.")

        # self.target = get_target(target) # Temporarily disable if causing issues
        self.target = target
        self.coupling_map = self.target.build_coupling_map()
        self.cnot_errors = {}
        self.swap_costs = {}
        self.gate_errors = {}
        self.readout_errors = {}
        self._build_noise_model()

    def _get_operation_error(self, op_name, qubits):
        """A generic helper to safely get error for any operation."""
        try:
            props = self.target[op_name][qubits]
            if props and props.error is not None:
                return props.error
            # Return a default high error if not specified
            return 0.1
        except (KeyError, AttributeError):
            # Return a default high error if gate/qubits not found
            return 0.1

    def _build_noise_model(self):
        if not self.coupling_map:
            return

        # Iterate through the qubits and get the error of the 'measure' op on each
        for q_idx in range(self.target.num_qubits):
            # Qubits for single-qubit ops are specified as a tuple, e.g., (0,)
            self.readout_errors[q_idx] = self._get_operation_error(
                "measure", (q_idx,)
            )

        for q1, q2 in self.coupling_map.get_edges():
            self.cnot_errors[(q1, q2)] = self._get_operation_error(
                "cx", (q1, q2)
            )

            fid_cx12 = 1.0 - self.cnot_errors.get((q1, q2), 0.1)
            # We need the error for the reverse CNOT for the SWAP cost
            fid_cx21 = 1.0 - self._get_operation_error("cx", (q2, q1))

            swap_fidelity = fid_cx12 * fid_cx21 * fid_cx12
            self.swap_costs[(q1, q2)] = 1.0 - swap_fidelity
            self.swap_costs[(q2, q1)] = 1.0 - swap_fidelity

    def get_correction_rules(self):
        """Returns a placeholder dictionary of correction rules."""
        return {}

    def get_hardware_vector(self) -> list[float]:
        """
        Calculates a fixed-length vector summarizing the device's noise.
        """
        cnot_errors_list = list(self.cnot_errors.values())
        if not cnot_errors_list:
            avg_cnot_error = 0.1
            std_cnot_error = 0.0
        else:
            avg_cnot_error = np.mean(cnot_errors_list)
            std_cnot_error = np.std(cnot_errors_list)

        readout_errors_list = list(self.readout_errors.values())
        if not readout_errors_list:
            avg_readout_error = 0.1
        else:
            avg_readout_error = np.mean(readout_errors_list)

        # Explicitly cast each numpy float to a standard Python float.
        # This makes the list JSON serializable.
        return [
            float(avg_cnot_error),
            float(std_cnot_error),
            float(avg_readout_error),
        ]

    def get_t1_t2(self, qubit: int) -> tuple[float, float]:
        """Safely gets T1 and T2 times for a qubit."""
        try:
            # NOTE: The exact access path depends on the Qiskit version and backend object.
            # This is for a modern `target` object.
            if not isinstance(qubit, int):
                qubit = (
                    qubit.index
                )  # or use another method to derive the index
            t1 = self.target.qubit_properties[qubit].t1
            t2 = self.target.qubit_properties[qubit].t2
            return (t1, t2)
        except (AttributeError, IndexError):
            # Return a very poor default value if data is missing
            return (0.0, 0.0)

    def get_readout_error(self, qubit: int) -> float:
        """Safely gets readout error for a qubit."""
        return self.readout_errors.get(qubit, 1.0)  # Default to max error

    def get_gate_properties(
        self, op_name: str, qubits: tuple
    ) -> tuple[float, float]:
        """Safely gets error and duration for a given gate on specific qubits."""
        try:
            props = self.target[op_name][qubits]
            error = props.error if (props and props.error is not None) else 1.0
            duration = (
                props.duration
                if (props and props.duration is not None)
                else 0.0
            )
            return (error, duration)
        except (KeyError, AttributeError):
            # Return a very poor default value if gate/qubits not supported
            return (1.0, 0.0)

    def get_gate_error(self, gate_name: str, physical_qubits: tuple) -> float:
        """
        Gets the error for any gate on a specific set of physical qubits.
        Handles looking up the correct key format.
        """
        # Ensure qubits are sorted for consistent dictionary key lookup
        qubits_key = tuple(sorted(physical_qubits))

        # --- Look for the specific gate and qubit combination ---
        error = self.gate_errors.get((gate_name, qubits_key))

        if error is not None:
            return error

        # --- Fallback Logic (Optional but Recommended) ---
        # If a specific gate (e.g., 'cx' from a user) isn't found, but the native
        # gate (e.g., 'ecr') exists for those qubits, use the native gate's error.
        # This handles cases where a circuit hasn't been fully translated yet.
        if len(qubits_key) == 2:
            native_2q_gate_name = next(
                (
                    name
                    for name, props in self.gate_errors.items()
                    if name[1] == qubits_key
                ),
                None,
            )
            if native_2q_gate_name:
                return self.gate_errors[native_2q_gate_name]

        # Return a sensible default if no error data is found at all.
        # A small non-zero value is better than zero.
        return 0.001


def get_target(backend_like: Union[Backend, Target]) -> Target:
    """
    Safely extracts a qiskit.transpiler.Target object from any backend-like source.

    This function provides a single, reliable interface to get a Target,
    handling modern BackendV2 objects, legacy-style BackendV1 objects, and
    raw Target objects transparently without using deprecated imports.

    Args:
        backend_like: An object representing a backend (V1, V2, or Target).

    Returns:
        A qiskit.transpiler.Target instance fully describing the backend.
    """
    if isinstance(backend_like, Target):
        # The object is already a Target.
        return backend_like

    if isinstance(backend_like, BackendV2):
        # Modern BackendV2 object: The target is a direct attribute.
        return backend_like.target

    if isinstance(backend_like, Backend):
        # Legacy BackendV1-style object: It's a Backend but not a BackendV2.
        # We must construct the Target from its configuration.
        config = backend_like.configuration()
        return Target.from_configuration(
            coupling_map=config.coupling_map,
            dt=config.dt,
            basis_gates=config.basis_gates,
        )

    raise TypeError(
        f"Unrecognized backend/target type: {type(backend_like)}. "
        "Expected a Qiskit Backend or Target instance."
    )


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


class CircuitCNN(nn.Module):
    """
    An efficient 1D Convolutional Neural Network to predict quantum circuit fidelity.
    Designed for high-speed inference.
    """

    def __init__(
        self,
        feature_dim: int,  # Dimensionality of the input feature vector (e.g., 16)
        model_dim: int,  # Number of channels in the CNN layers (e.g., 64 or 128)
        n_layers: int,  # Number of convolutional blocks (e.g., 3 or 4)
        kernel_size: int = 3,  # Size of the sliding window (3 is a great default)
        dropout: float = 0.1,
    ):
        super().__init__()

        # The CNN expects input of shape [batch, channels, length].
        # Our data is [batch, length, channels], so we will permute it.

        self.conv_layers = nn.ModuleList()
        # The first layer projects the input feature_dim to the model_dim
        in_channels = feature_dim
        out_channels = model_dim

        for i in range(n_layers):
            # Each block consists of a convolution, activation, and dropout
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",  # 'same' padding keeps the sequence length constant
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv_layers.append(conv_block)
            # The input channels for the next layer is the output of this one
            in_channels = out_channels

        # The classifier head takes the flattened output of the CNN
        # The input size to the linear layer depends on the final number of channels
        # and the sequence length after pooling.
        self.classifier_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feature_dim]
        """
        # Permute the input to match Conv1d's expected shape: [batch, channels, length]
        x = src.permute(0, 2, 1)

        # Pass through the convolutional layers
        for conv_block in self.conv_layers:
            x = conv_block(x)

        # Global Average Pooling: Take the mean across the entire sequence length dimension.
        # This creates a single feature vector for the whole circuit.
        # Input x has shape [batch, model_dim, seq_len]
        # Output x_pooled has shape [batch, model_dim]
        x_pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Pass the final feature vector through the classifier
        prediction = self.classifier_head(x_pooled)

        return prediction


class CircuitFNO(nn.Module):
    """
    A 1D Fourier Neural Operator model to predict quantum circuit fidelity.
    This model learns in the frequency domain, which is a natural fit for
    wave-like quantum dynamics.
    """

    def __init__(
        self,
        feature_dim: int,  # Input dimension (e.g., 16)
        n_modes: int,  # Number of Fourier modes to keep. This is the key hyperparameter.
        hidden_channels: int,  # The "width" of the FNO layers
        n_layers: int,  # Number of FNO blocks
        dropout: float = 0.1,
    ):
        super().__init__()

        # The FNO1D model from the neuraloperator library does all the hard work.
        self.fno = FNO(
            n_modes=(n_modes,),  # Must be a tuple for 1D data
            hidden_channels=hidden_channels,
            in_channels=feature_dim,
            out_channels=1,  # We want to directly output a single value (logit)
            n_layers=n_layers,
            use_mlp=True,  # Adds a small MLP after the spectral convolution
            mlp_dropout=dropout,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feature_dim]
        """
        # FNO expects input of shape [batch, channels, length]
        x = src.permute(0, 2, 1)

        # Pass through the FNO layers
        x = self.fno(x)

        # The FNO output is [batch, 1, seq_len]. We need to pool it.
        # Global Average Pooling to get a single value for the circuit.
        x_pooled = torch.mean(x, dim=2)  # Shape: [batch, 1]

        # Apply the final Sigmoid activation to get a probability
        prediction = torch.sigmoid(x_pooled)

        return prediction


# ==============================================================================
# 2. DATASET CLASS (Copied directly into this file)
# ==============================================================================
class FidelityDataset(Dataset):
    """
    A flexible PyTorch Dataset that works with pre-computed feature tensors.
    It can generate data for either a Transformer or a Graph Neural Network model
    from the SAME input data source.
    """

    def __init__(
        self,
        raw_data,  # List of {'feature_tensor': [...], 'fidelity_label': ...}
        model_type: str,  # Either "transformer" or "gnn"
        max_seq_len: int,  # Required for Transformer padding
        feature_dim: int,  # The dimension of the feature vectors
    ):
        self.data = raw_data
        self.model_type = model_type
        self.max_len = max_seq_len
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        feature_sequence = item["feature_tensor"]
        fidelity_label = torch.tensor(
            [item["fidelity_label"]], dtype=torch.float32
        )

        if (
            self.model_type == "transformer"
            or self.model_type == "cnn"
            or self.model_type == "fno"
        ):
            # --- Generate padded tensor for the Transformer ---

            # 1. Truncate if too long
            if len(feature_sequence) > self.max_len:
                feature_sequence = feature_sequence[: self.max_len]

            # 2. Pad with zeros if too short
            padding_needed = self.max_len - len(feature_sequence)
            if padding_needed > 0:
                zero_vector = [0.0] * self.feature_dim
                feature_sequence.extend([zero_vector] * padding_needed)

            final_data = torch.tensor(feature_sequence, dtype=torch.float32)
            return final_data, fidelity_label

        elif self.model_type == "gnn":
            # --- Generate a graph Data object for the GNN ---

            # Node features are just the sequence from the dataset
            node_features = torch.tensor(feature_sequence, dtype=torch.float32)

            # Create a simple sequential edge_index: 0->1, 1->2, 2->3, ...
            num_nodes = node_features.shape[0]
            if num_nodes > 1:
                source_nodes = torch.arange(0, num_nodes - 1)
                dest_nodes = torch.arange(1, num_nodes)
                edge_index = torch.stack([source_nodes, dest_nodes], dim=0)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            final_data = Data(
                x=node_features, edge_index=edge_index, y=fidelity_label
            )
            return final_data, fidelity_label


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
        help="model transformer, gnn, cnn, fno",
    )

    args = parser.parse_args()

    # --- Setup ---
    if args.model not in ["transformer", "gnn", "cnn", "fno"]:
        raise ValueError(
            "model_type must be either 'transformer' or 'gnn' or 'cnn' or 'fno'"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # --- Load Data ---
    print(f"Loading dataset from: {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        raw_data = json.load(f)

    # 2. Create the appropriate dataset instance
    #    The FidelityDataset is now smart enough to handle both cases.
    print(f"Preparing data for '{args.model}' model...")
    target = FakeWashingtonV2()
    noise_profile = DeviceNoiseProfile(get_target(target))

    full_dataset = FidelityDataset(
        raw_data=raw_data,
        model_type=args.model,
        max_seq_len=args.max_seq_len,
        feature_dim=args.feature_dim,
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

    elif args.model == "cnn" or args.model == "fno":
        print("Initializing CircuitCNN (1D-CNN) model...")
        model = CircuitCNN(
            feature_dim=args.feature_dim,
            model_dim=args.model_dim,  # For a CNN, you can often use a smaller dim, e.g., 64
            n_layers=args.n_layers,  # A few layers (e.g., 3-4) is usually enough
            dropout=0.1,
        ).to(device)

        # Use the STANDARD PyTorch DataLoader, same as the Transformer
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
    elif args.model == "fno":
        print("Initializing CircuitCNN (1D-CNN) model...")
        model = CircuitFNO(
            feature_dim=args.feature_dim,
            model_dim=args.model_dim,  # For a CNN, you can often use a smaller dim, e.g., 64
            n_layers=args.n_layers,  # A few layers (e.g., 3-4) is usually enough
            dropout=0.1,
        ).to(device)

        # Use the STANDARD PyTorch DataLoader, same as the Transformer
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

    else:
        raise ValueError(f"Unknown model type specified: {args.model}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(
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

        for batch in train_pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                if (
                    args.model == "transformer"
                    or args.model == "cnn"
                    or args.model == "fno"
                ):
                    # Transformer and CNN use the standard DataLoader and expect a tuple
                    features, labels = batch
                    features, labels = (
                        features.to(device, non_blocking=True),
                        labels.to(device, non_blocking=True),
                    )
                    outputs = model(features)

                elif args.model == "gnn":
                    # GNN uses the GeometricDataLoader and expects a single Batch object
                    batch_data = batch.to(device)
                    labels = batch_data.y
                    outputs = model(batch_data)

                loss = criterion(outputs, labels)

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
            for batch in val_loader:
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    if (
                        args.model == "transformer"
                        or args.model == "cnn"
                        or args.model == "fno"
                    ):
                        features, labels = batch
                        features, labels = (
                            features.to(device, non_blocking=True),
                            labels.to(device, non_blocking=True),
                        )
                        outputs = model(features)
                    else:  # GNN case
                        batch_data = batch.to(device)
                        labels = batch_data.y
                        outputs = model(batch_data)

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
