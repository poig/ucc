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


# ==============================================================================
# 2. DATASET CLASS (Copied directly into this file)
# ==============================================================================
def load_and_deserialize_data(path):
    print(f"Loading and deserializing dataset from: {path}")
    with open(path, "r") as f:
        # For now, we assume the json contains a placeholder for the DAG
        # In a real scenario, you'd use qiskit.qpy to load circuits
        raw_data = json.load(f)
        # THIS IS A CRITICAL STEP YOU MUST IMPLEMENT
        # For now, we'll crash if the data is not in the right format
        for item in raw_data:
            if "dag" not in item:
                raise KeyError("Dataset entry is missing 'dag' key.")
            # You would have a function here like: item['dag'] = dag_from_your_format(item['dag'])
    return raw_data


class FidelityDataset(Dataset):
    """
    A flexible PyTorch Dataset that can generate data for either a Transformer
    or a Graph Neural Network model.

    It can operate in two modes:
    1. On-the-fly processing: If `raw_data` is provided, it converts DAGs
       to features with every __getitem__ call (slower).
    2. Pre-processed loading: If `preprocessed_path` is provided, it loads
       a pre-saved list of processed data objects (much faster).
    """

    def __init__(
        self,
        model_type: str,
        # --- Data Sources (provide ONE of these) ---
        raw_data=None,
        preprocessed_path: str = None,
        # --- Config for On-the-fly Processing ---
        noise_profile=None,
        max_seq_len: int = 1024,
        feature_dim: int = 16,
        gate_vocab: list = None,
        # ... other feature config ...
    ):
        if model_type not in ["transformer", "gnn"]:
            raise ValueError(
                "model_type must be either 'transformer' or 'gnn'"
            )

        if raw_data is None and preprocessed_path is None:
            raise ValueError(
                "Must provide either 'raw_data' or 'preprocessed_path'"
            )

        if raw_data is not None and preprocessed_path is not None:
            print(
                "Warning: Both raw_data and preprocessed_path provided. Using pre-processed data."
            )

        self.model_type = model_type

        # --- THE NEW LOGIC ---
        if preprocessed_path and os.path.exists(preprocessed_path):
            print(f"Loading pre-processed data from: {preprocessed_path}")
            self.data = torch.load(preprocessed_path)
            self.is_preprocessed = True
        else:
            print("Processing raw data on-the-fly.")
            if raw_data is None:
                raise FileNotFoundError(
                    f"Pre-processed file not found at: {preprocessed_path}"
                )
            self.data = raw_data
            self.is_preprocessed = False
            # Store config needed for on-the-fly processing
            self.noise_profile = noise_profile
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
            # ... store other feature configs ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.is_preprocessed:
            final_data = item["data"]
            fidelity_label = item["fidelity_label"]
        else:
            dag = item["dag"]
            fidelity_label = torch.tensor(
                [item["fidelity_label"]], dtype=torch.float32
            )

            if self.model_type == "transformer":
                feature_sequence = self._dag_to_feature_sequence(dag)
                # Padding & Truncation
                if len(feature_sequence) > self.max_len:
                    feature_sequence = feature_sequence[: self.max_len]
                padding_needed = self.max_len - len(feature_sequence)
                if padding_needed > 0:
                    zero_vector = [0.0] * self.feature_dim
                    feature_sequence.extend([zero_vector] * padding_needed)
                final_data = torch.tensor(
                    feature_sequence, dtype=torch.float32
                )

            elif self.model_type == "gnn":
                final_data = physical_dag_to_graph(
                    dag, self.noise_profile, ...
                )  # Pass configs
                final_data.y = fidelity_label

        # For GNN, the data and label are bundled. For consistency, we return both.
        if self.model_type == "gnn":
            return final_data, final_data.y
        else:
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
    raw_data = load_and_deserialize_data(args.dataset_path)
    # with open(args.dataset_path, "r") as f:
    #     raw_data = json.load(f)

    # 2. Create the appropriate dataset instance
    #    The FidelityDataset is now smart enough to handle both cases.
    print(f"Preparing data for '{args.model}' model...")
    target = FakeWashingtonV2()
    noise_profile = DeviceNoiseProfile(target)

    GATE_VOCAB = ["cx", "sx", "rz", "x", "id", "measure", "other"]

    full_dataset = FidelityDataset(
        raw_data=raw_data,
        noise_profile=noise_profile,
        model_type=args.model,
        max_seq_len=args.max_seq_len,
        feature_dim=args.feature_dim,
        gate_vocab=GATE_VOCAB,
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
                if args.model == "transformer":
                    features, labels = batch
                    features, labels = (
                        features.to(device, non_blocking=True),
                        labels.to(device, non_blocking=True),
                    )
                    outputs = model(features)
                elif args.model == "gnn":
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
                    if args.model == "transformer":
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
