#!/usr/bin/env python
import torch
import math
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import TranspilerError, Layout
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import SwapGate
from torch.cuda.amp import autocast
from qiskit import QuantumRegister


class MLFidelityRouter(TransformationPass):
    """
    An AI-driven routing pass that uses a Transformer model to predict the fidelity
    of different routing decisions, guided by just-in-time calibration data.
    This version uses a correct, constructive, state-preserving algorithm.
    """

    # ===================================================================
    # __init__ and Feature Extractor (No changes needed here)
    # ===================================================================
    def __init__(
        self, target, model, noise_profile, max_seq_len: int, config=None
    ):
        super().__init__()
        self.target = target
        self.model = model
        self.noise_profile = noise_profile
        self.coupling_map = target.build_coupling_map()
        self.dist_matrix = self.coupling_map.distance_matrix
        self.config = config or {}
        self.candidate_top_k = self.config.get("candidate_top_k", 5)
        self.full_layout_mode = self.config.get("full_layout", True)
        if self.model:
            self.model.eval()
        self.GATE_VOCAB = ["cx", "sx", "rz", "x", "id", "measure", "other"]
        self.NUM_PARAMS = 1
        self.NUM_QUBIT_FEATURES = 3
        self.NUM_GATE_CAL_FEATURES = 2
        self.FEATURE_DIM = (
            len(self.GATE_VOCAB)
            + self.NUM_PARAMS
            + self.NUM_QUBIT_FEATURES * 2
            + self.NUM_GATE_CAL_FEATURES
        )
        self.MAX_LEN = max_seq_len

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # --- 1. SETUP AND LAYOUT HANDLING ---
        if (
            "layout" not in self.property_set
            or self.property_set["layout"] is None
        ):
            # Create a basic trivial layout if none was provided.
            initial_layout = self._create_initial_layout(dag)
        else:
            # Use the layout from the previous pass (e.g., SabreLayout)
            initial_layout = self.property_set["layout"]

        # Convert the partial layout into a full layout that includes idle qubits.
        layout = self._ensure_full_layout(initial_layout)

        # This new circuit's virtual qubits [0...N-1] directly represent physical qubits.
        new_dag = self._create_physical_dag()

        nodes = list(dag.topological_op_nodes())
        node_idx = 0

        # --- 2. MAIN ROUTING LOOP ---
        while node_idx < len(nodes):
            node = nodes[node_idx]

            # This logic is now safe because our `layout` is full.
            physical_qargs = [layout[vq] for vq in node.qargs]

            if node.op.num_qubits != 2:
                self._apply_gate_physical(
                    new_dag, node.op, physical_qargs, node.cargs
                )
                node_idx += 1
                continue

            p_q0, p_q1 = physical_qargs[0], physical_qargs[1]
            if self._is_physically_connected(p_q0, p_q1):
                self._apply_gate_physical(
                    new_dag, node.op, physical_qargs, node.cargs
                )
                node_idx += 1
            else:
                # Gate is blocked, must SWAP
                candidates = self._generate_swap_candidates(p_q0, p_q1)
                if not candidates:
                    raise TranspilerError(
                        f"No valid swaps for gate {node.op.name}."
                    )

                remaining_nodes = nodes[node_idx:]
                # Using a stable heuristic for SWAP choice.

                best_swap = self._find_best_ai_swap(
                    candidates, new_dag, remaining_nodes, layout
                )

                self._apply_swap_physical(new_dag, best_swap)

                # Now this call is safe because the layout knows about all physical qubits.
                layout.swap(best_swap[0], best_swap[1])

        self.property_set["layout"] = layout
        return new_dag

    def _create_initial_layout(self, dag: DAGCircuit) -> Layout:
        """Creates a trivial initial layout mapping virtual q[i] to physical i."""
        num_virtual_qubits = len(dag.qubits)
        if num_virtual_qubits > self.target.num_qubits:
            raise TranspilerError(
                f"Cannot map {num_virtual_qubits} qubits to {self.target.num_qubits} physical qubits."
            )

        # Create a trivial mapping {virtual_qubit_object: physical_index}
        v2p_map = {vq: i for i, vq in enumerate(dag.qubits)}
        return Layout(v2p_map)

    def _apply_gate_physical(
        self, dag: DAGCircuit, op, p_qargs: list, c_cargs: list
    ):
        """Applies a gate to the given physical qubit indices in the new_dag."""
        # The virtual qubits of new_dag directly correspond to physical indices
        dag.apply_operation_back(
            op, qargs=[dag.qubits[p] for p in p_qargs], cargs=c_cargs
        )

    def _apply_swap_physical(self, dag: DAGCircuit, physical_qubits: tuple):
        """Applies a SWAP gate to the given physical qubit indices."""
        p1, p2 = physical_qubits
        dag.apply_operation_back(
            SwapGate(), qargs=(dag.qubits[p1], dag.qubits[p2])
        )

    def _get_heuristic_score(
        self, swap_candidate: tuple, p_q0: int, p_q1: int, layout: Layout
    ) -> int:
        """Calculates the cheap heuristic score (distance) for a potential swap."""
        p_s1, p_s2 = swap_candidate

        # Simulate the swap to see where the target qubits would end up
        temp_p_q0, temp_p_q1 = p_q0, p_q1
        if p_q0 == p_s1:
            temp_p_q0 = p_s2
        elif p_q0 == p_s2:
            temp_p_q0 = p_s1
        if p_q1 == p_s1:
            temp_p_q1 = p_s2
        elif p_q1 == p_s2:
            temp_p_q1 = p_s1

        return self.dist_matrix[temp_p_q0][temp_p_q1]

    def _find_best_heuristic_swap(
        self, candidates: list, p_q0: int, p_q1: int
    ) -> tuple:
        """A simple heuristic to choose a SWAP that brings qubits closer."""
        best_swap = None
        min_dist = self.dist_matrix[p_q0][p_q1]

        for p_s1, p_s2 in candidates:
            # What would the new distance be if we swapped?
            # Simulate swapping p_q0 if it's one of the swap partners
            temp_p_q0 = p_q0
            if p_q0 == p_s1:
                temp_p_q0 = p_s2
            elif p_q0 == p_s2:
                temp_p_q0 = p_s1

            temp_p_q1 = p_q1
            if p_q1 == p_s1:
                temp_p_q1 = p_s2
            elif p_q1 == p_s2:
                temp_p_q1 = p_s1

            new_dist = self.dist_matrix[temp_p_q0][temp_p_q1]

            if new_dist < min_dist:
                min_dist = new_dist
                best_swap = (p_s1, p_s2)

        return best_swap if best_swap is not None else candidates[0]

    def _fill_layout_with_idle(self, layout: Layout):
        """
        Ensures a layout object has entries for all physical qubits on the device,
        marking unused ones as idle. THIS IS THE FINAL FIX for the KeyError.
        """
        # Get all physical qubits on the device
        physical_qubits = list(range(self.target.num_qubits))

        # Use the public method to add idle qubits. This correctly updates
        # the internal state of the Layout object.
        layout.add_idle_wires(physical_qubits)

    def _find_best_ai_swap(
        self,
        candidates: list,
        dag_so_far: DAGCircuit,
        remaining_nodes: list,
        layout: Layout,
    ) -> tuple:
        best_swap = None
        best_fidelity = -1.0

        for swap_candidate in candidates:
            # 1. Create a temporary "what-if" universe
            temp_dag = dag_so_far.copy_empty_like()
            temp_dag.compose(dag_so_far)
            temp_layout = layout.copy()

            # 2. Apply the candidate SWAP in the temporary universe
            self._apply_swap_physical(temp_dag, swap_candidate)
            temp_layout.swap(swap_candidate[0], swap_candidate[1])

            # 3. Build a "mock" version of the rest of the circuit
            #    This is a simplification: we assume no more swaps are needed after this one.
            #    This allows us to create a full circuit to evaluate.
            for node in remaining_nodes:
                physical_qargs = [temp_layout[vq] for vq in node.qargs]
                self._apply_gate_physical(
                    temp_dag, node.op, physical_qargs, node.cargs
                )

            # 4. Evaluate this entire hypothetical circuit with the AI
            #    This requires a new feature extractor that takes a physical DAG and the noise profile.
            feature_tensor = self._physical_dag_to_feature_tensor(temp_dag)
            predicted_fidelity = self._evaluate_single_state_with_ai(
                feature_tensor
            )

            # 5. Keep track of the SWAP that leads to the best predicted outcome
            if predicted_fidelity > best_fidelity:
                best_fidelity = predicted_fidelity
                best_swap = swap_candidate

        return best_swap if best_swap is not None else candidates[0]

    def _create_physical_dag(self) -> DAGCircuit:
        """Creates a new DAG with virtual qubits matching the physical device size."""
        new_dag = DAGCircuit()
        device_qreg = QuantumRegister(self.target.num_qubits, "q")
        new_dag.add_qreg(device_qreg)
        return new_dag

    def _ensure_full_layout(self, layout: Layout) -> Layout:
        """
        Takes a Layout and ensures it has an entry for every physical qubit.

        If a physical qubit is not in the layout, it's considered idle
        and will be added with a mapping to `None`.

        Args:
            layout: The (potentially partial) Layout object.

        Returns:
            A new, "full" Layout object that is safe to use for swapping.
        """
        # Get the dictionary of active virtual-to-physical mappings
        v2p_map = layout.get_virtual_bits()

        # Create a new layout object from this map. This is the safest way
        # to ensure we have a clean copy to work with.
        full_layout = Layout(v2p_map)

        # --- The "Fill in the Blanks" Logic ---

        # Find all physical qubits that are not being used by active virtual qubits.
        all_physical_qubits = set(range(self.target.num_qubits))
        active_physical_qubits = set(full_layout.get_physical_bits())
        idle_physical_qubits = all_physical_qubits - active_physical_qubits

        # Manually add the idle qubits to the layout's internal physical-to-virtual dictionary.
        # This guarantees that every physical qubit index is a valid key.
        for p_idle in idle_physical_qubits:
            full_layout._p2v[p_idle] = None

        return full_layout

    def _add_swap_gate(
        self, dag: DAGCircuit, physical_qubits: tuple, layout: Layout
    ):
        """
        A 'stateless' helper that only adds a SWAP gate to a DAG based on a layout.
        It does NOT modify the layout object itself.
        """
        p1, p2 = physical_qubits
        # Use the layout's internal dictionary for robustness.
        v1 = layout._p2v[p1]
        v2 = layout._p2v[p2]
        dag.apply_operation_back(SwapGate(), qargs=(v1, v2))

    def _generate_swap_candidates(self, p_q0: int, p_q1: int) -> list:
        candidates = set()
        for neighbor in self.coupling_map.neighbors(p_q0):
            candidates.add(tuple(sorted((p_q0, neighbor))))
        for neighbor in self.coupling_map.neighbors(p_q1):
            candidates.add(tuple(sorted((p_q1, neighbor))))
        return list(candidates)[: self.candidate_top_k]

    def _apply_gate(self, dag: DAGCircuit, node):
        dag.apply_operation_back(node.op, node.qargs, node.cargs)

    def _create_empty_dag_like(self, dag: DAGCircuit) -> DAGCircuit:
        """Creates a new DAG with virtual qubits matching the physical device size."""
        new_dag = DAGCircuit()
        # Create a new QuantumRegister that represents the entire physical device
        device_qreg = QuantumRegister(self.target.num_qubits, "q")
        new_dag.add_qreg(device_qreg)
        # Add any classical registers from the original circuit
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        return new_dag

    def _make_layout_full(self, partial_layout: Layout) -> Layout:
        """Takes a partial layout and returns a new, full layout for the device."""
        # Start with a copy of the mapping from the partial layout.
        v2p_map = partial_layout.get_virtual_bits()
        full_layout = Layout(v2p_map)

        # Find all physical qubits that are not being used.
        all_physical_qubits = set(range(self.target.num_qubits))
        active_physical_qubits = set(full_layout.get_physical_bits())
        idle_physical_qubits = all_physical_qubits - active_physical_qubits

        # Manually add the idle qubits to the layout's internal dictionary.
        for p_idle in idle_physical_qubits:
            full_layout._p2v[p_idle] = None

        return full_layout

    def _is_physically_connected(self, p_q0: int, p_q1: int) -> bool:
        return self.coupling_map.graph.has_edge(p_q0, p_q1)

    def _evaluate_single_state_with_ai(self, feature_tensor):
        if not self.model:
            return 0.5
        model_device = next(self.model.parameters()).device
        input_tensor = feature_tensor.unsqueeze(0).to(model_device)
        with torch.no_grad(), autocast():
            prediction = self.model(input_tensor)
        return prediction.item()

    def _physical_dag_to_feature_tensor(
        self, physical_dag: DAGCircuit
    ) -> torch.Tensor:
        """
        The "Feature Extractor". Converts a physically-mapped DAGCircuit into a
        tensor representation for the CircuitFormer model.

        For each gate, it creates a rich feature vector containing information about
        the gate's type, parameters, and the JIT calibration data of the
        physical qubits it acts upon.

        Args:
            physical_dag: The circuit's DAG representation, where dag.qubits[i]
                          corresponds to physical qubit i.

        Returns:
            A PyTorch tensor of shape [self.MAX_LEN, self.FEATURE_DIM].
        """
        gate_feature_sequence = []

        # Iterate through all operations in a topological order
        for node in physical_dag.op_nodes():
            # --- 1. Gate Type Encoding (One-Hot) ---
            gate_type_encoding = [0.0] * len(self.GATE_VOCAB)
            op_name = node.op.name
            if op_name in self.GATE_VOCAB:
                gate_type_encoding[self.GATE_VOCAB.index(op_name)] = 1.0
            else:
                gate_type_encoding[-1] = 1.0  # 'other' category

            # --- 2. Gate Parameter Encoding ---
            gate_params = [0.0] * self.NUM_PARAMS
            if hasattr(node.op, "params") and node.op.params:
                # Normalize angle for better ML stability
                gate_params[0] = float(node.op.params[0]) / (2 * math.pi)

            # --- 3. Get Physical Qubit Indices ---
            # This is now much simpler. The qubit's index in the DAG IS the physical index.
            physical_indices = [
                physical_dag.find_bit(q).index for q in node.qargs
            ]

            # --- 4. Physical Qubit and Gate Calibration Features ---
            phys_q1_features = [0.0] * self.NUM_QUBIT_FEATURES
            phys_q2_features = [0.0] * self.NUM_QUBIT_FEATURES
            gate_cal_features = [0.0] * self.NUM_GATE_CAL_FEATURES

            if physical_indices:
                pq1 = physical_indices[0]
                # Get features for the first physical qubit from the noise profile
                t1, t2 = self.noise_profile.get_t1_t2(pq1)
                readout_err = self.noise_profile.get_readout_error(pq1)
                phys_q1_features = [t1, t2, readout_err]

                if len(physical_indices) == 2:
                    # It's a 2-qubit gate
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
                else:
                    # It's a 1-qubit gate
                    gate_err, gate_dur = (
                        self.noise_profile.get_gate_properties(op_name, (pq1,))
                    )
                    gate_cal_features = [gate_err, gate_dur]

            # --- 5. Assemble the Final Feature Vector ---
            feature_vector = (
                gate_type_encoding
                + gate_params
                + phys_q1_features
                + phys_q2_features
                + gate_cal_features
            )
            gate_feature_sequence.append(feature_vector)

        # --- 6. Padding and Truncation ---
        # Truncate if longer than MAX_LEN
        truncated_sequence = gate_feature_sequence[: self.MAX_LEN]

        # Pad the sequence with zero vectors if it's too short
        padding_needed = self.MAX_LEN - len(truncated_sequence)
        if padding_needed > 0:
            zero_vector = [0.0] * self.FEATURE_DIM
            padded_sequence = (
                truncated_sequence + [zero_vector] * padding_needed
            )
        else:
            padded_sequence = truncated_sequence

        # --- 7. Convert to PyTorch Tensor ---
        return torch.tensor(padded_sequence, dtype=torch.float32)
