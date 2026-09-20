"""Training-only residual projectors for RSIAT and QG-ORS ablations.

The quantum dependency is intentionally lazy: importing or training the RSIAT
baseline does not require PennyLane.
"""

import math

import torch
from torch import nn

from utils.toolkit import AutoencoderSigmoid


PROJECTOR_TYPES = {
    "rsiat",
    "scalar",
    "classical",
    "quantum_frozen",
    "quantum",
}


class ClassicalControl(nn.Module):
    """Parameter-light classical control with the same input/output width as a VQC."""

    def __init__(self, input_dims, control_dims, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or 2 * control_dims
        self.compress = nn.Linear(input_dims, control_dims)
        self.norm = nn.LayerNorm(control_dims)
        self.controller = nn.Sequential(
            nn.Linear(control_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, control_dims),
        )

    def forward(self, x):
        angles = math.pi * torch.tanh(self.norm(self.compress(x)))
        return self.controller(angles)


class QuantumControl(nn.Module):
    """Compress a feature and produce Pauli-Z expectation values with a VQC."""

    def __init__(
        self,
        input_dims,
        num_qubits=4,
        num_layers=2,
        backend="default.qubit",
        trainable=True,
    ):
        super().__init__()
        if num_qubits < 2:
            raise ValueError("QG-ORS requires at least two qubits for ring entanglement.")
        if num_layers < 1:
            raise ValueError("QG-ORS requires at least one variational layer.")

        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "projector_type={!r} requires PennyLane. Install the optional "
                "QG-ORS dependencies from requirements-qgors.txt."
                .format("quantum" if trainable else "quantum_frozen")
            ) from exc

        self.num_qubits = int(num_qubits)
        self.num_layers = int(num_layers)
        self.backend = backend
        self.compress = nn.Linear(input_dims, self.num_qubits)
        self.norm = nn.LayerNorm(self.num_qubits)

        device = qml.device(backend, wires=self.num_qubits)
        # ``default.qubit`` executes on the host CPU. Keep its TorchLayer on
        # CPU even when the surrounding vision model is moved to CUDA.
        self._simulator_device = (
            torch.device("cpu") if backend == "default.qubit" else None
        )
        wires = tuple(range(self.num_qubits))

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Initial angle encoding followed by variational layers and data
            # re-uploading. The final layer does not need another upload.
            qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
            for layer_index in range(self.num_layers):
                for wire in wires:
                    qml.RY(weights[layer_index, wire, 0], wires=wire)
                    qml.RZ(weights[layer_index, wire, 1], wires=wire)
                for wire in wires:
                    qml.CNOT(wires=[wire, (wire + 1) % self.num_qubits])
                if layer_index + 1 < self.num_layers:
                    qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
            return [qml.expval(qml.PauliZ(wire)) for wire in wires]

        weight_shapes = {
            "weights": (self.num_layers, self.num_qubits, 2),
        }
        self.quantum_layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
            init_method=nn.init.uniform_,
        )
        if not trainable:
            for parameter in self.quantum_layer.parameters():
                parameter.requires_grad = False

    def ensure_simulator_device(self):
        """Move quantum_layer back to the simulator device (CPU) if needed.

        Call this after any `.to(cuda)` on a parent module to undo the
        blanket device move that PyTorch applies to all sub-modules.
        """
        if self._simulator_device is not None:
            self.quantum_layer.to(self._simulator_device)

    def forward(self, x):
        angles = math.pi * torch.tanh(self.norm(self.compress(x)))
        quantum_input = angles
        if self._simulator_device is not None:
            # Ensure quantum_layer lives on the simulator device (CPU).
            self.ensure_simulator_device()
            quantum_input = angles.to(self._simulator_device)

        # Some PennyLane devices promote values to float64. Restore both the
        # feature device and dtype before the classical projection.
        return self.quantum_layer(quantum_input).to(
            device=x.device,
            dtype=x.dtype,
        )


class ScalarGatedAutoencoder(AutoencoderSigmoid):
    """RSIAT residual projector with one learned, input-independent scale."""

    def __init__(self, input_dims=768, code_dims=384, gate_amplitude=0.5):
        super().__init__(input_dims=input_dims, code_dims=code_dims)
        _validate_gate_amplitude(gate_amplitude)
        self.gate_amplitude = float(gate_amplitude)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward_with_aux(self, x):
        latent = self.encoder(x)
        scale = 1.0 + self.gate_amplitude * torch.tanh(self.gate_logit)
        residual = self.decoder(scale * latent)
        projected = x + residual
        expanded_scale = scale.expand_as(latent)
        return projected, _projector_aux(latent, expanded_scale, residual)

    def forward(self, x):
        return self.forward_with_aux(x)[0]


class AdaptiveGatedAutoencoder(AutoencoderSigmoid):
    """Latent-gated RSIAT projector shared by classical and quantum controls."""

    def __init__(
        self,
        input_dims=768,
        code_dims=384,
        control_dims=4,
        gate_amplitude=0.5,
        control_type="classical",
        num_layers=2,
        backend="default.qubit",
        classical_hidden_dims=None,
    ):
        super().__init__(input_dims=input_dims, code_dims=code_dims)
        _validate_gate_amplitude(gate_amplitude)
        self.gate_amplitude = float(gate_amplitude)

        if control_type == "classical":
            self.control = ClassicalControl(
                input_dims,
                control_dims,
                hidden_dims=classical_hidden_dims,
            )
        elif control_type in {"quantum", "quantum_frozen"}:
            self.control = QuantumControl(
                input_dims,
                num_qubits=control_dims,
                num_layers=num_layers,
                backend=backend,
                trainable=control_type == "quantum",
            )
        else:
            raise ValueError("Unknown QG-ORS control type: {}".format(control_type))

        self.gate_projection = nn.Linear(control_dims, code_dims)
        # This makes scale == 1 exactly at initialization, so RSIAT is a strict
        # special case and the first forward pass is baseline-equivalent.
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)

    def ensure_quantum_cpu(self):
        """Move quantum sub-layers back to CPU after a blanket .to(cuda)."""
        if hasattr(self.control, "ensure_simulator_device"):
            self.control.ensure_simulator_device()

    def forward_with_aux(self, x):
        latent = self.encoder(x)
        control = self.control(x)
        scale = 1.0 + self.gate_amplitude * torch.tanh(
            self.gate_projection(control)
        )
        residual = self.decoder(scale * latent)
        projected = x + residual
        return projected, _projector_aux(latent, scale, residual, control)

    def forward(self, x):
        return self.forward_with_aux(x)[0]


def _validate_gate_amplitude(gate_amplitude):
    if not 0.0 <= float(gate_amplitude) <= 1.0:
        raise ValueError(
            "qg_gate_amplitude must be in [0, 1] so the latent scale remains non-negative."
        )


def _projector_aux(latent, scale, residual, control=None):
    aux = {
        "latent": latent,
        "scale": scale,
        "residual": residual,
    }
    if control is not None:
        aux["control"] = control
    return aux


def build_projector(args, input_dims=768):
    """Build an RSIAT-compatible projector from an experiment config."""
    projector_type = str(args.get("projector_type", "rsiat")).lower()
    if projector_type not in PROJECTOR_TYPES:
        raise ValueError(
            "Unknown projector_type {!r}; expected one of {}."
            .format(projector_type, sorted(PROJECTOR_TYPES))
        )

    common = {
        "input_dims": input_dims,
        "code_dims": int(args["ae_code_dims"]),
    }
    if projector_type == "rsiat":
        return AutoencoderSigmoid(**common)

    gate_amplitude = float(args.get("qg_gate_amplitude", 0.5))
    if projector_type == "scalar":
        return ScalarGatedAutoencoder(
            **common,
            gate_amplitude=gate_amplitude,
        )

    return AdaptiveGatedAutoencoder(
        **common,
        control_dims=int(args.get("qg_num_qubits", 4)),
        gate_amplitude=gate_amplitude,
        control_type=projector_type,
        num_layers=int(args.get("qg_num_layers", 2)),
        backend=args.get("qg_backend", "default.qubit"),
        classical_hidden_dims=args.get("qg_classical_hidden_dims"),
    )
