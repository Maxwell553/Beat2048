"""Search-free fusion of current-board convolutional and sparse neural policies."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .deep_rl import ConvAgent, FusionPolicy, encode
from .env import legal_actions, pack
from .rl import RLAgent


class FusionAgent:
    """Run the frozen neural fusion policy described by a checked manifest."""

    def __init__(self, manifest="models/rl/best_policy.json", device=None):
        manifest_path = Path(manifest)
        specification = json.loads(manifest_path.read_text())
        if specification.get("mode") != "fusion":
            raise ValueError("checkpoint manifest does not describe a fusion policy")
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        root = manifest_path.parent

        def checked(entry):
            path = Path(entry["checkpoint"])
            if not path.is_absolute():
                path = root / path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {path}")
            return path

        cnn_path = checked(specification["cnn"])
        fusion_entries = specification["fusion"]
        if isinstance(fusion_entries, dict):
            fusion_entries = [fusion_entries]
        fusion_paths = [checked(entry) for entry in fusion_entries]
        sparse_paths = [checked(entry) for entry in specification["sparse"]]
        expert_paths = [checked(entry) for entry in specification.get("experts", [])]
        self.cnn = ConvAgent(cnn_path, self.device)
        self.sparse = [RLAgent(path) for path in sparse_paths]
        self.experts = [ConvAgent(path, self.device) for path in expert_paths]
        self.models = []
        available_inputs = 296 + 4 * len(self.sparse) + 32 * len(self.experts)
        for fusion_path in fusion_paths:
            artifact = torch.load(
                fusion_path, map_location=self.device, weights_only=True
            )
            metadata = artifact["metadata"]
            if metadata["inputs"] > available_inputs:
                raise ValueError(
                    "fusion input size does not match its component manifest"
                )
            model = FusionPolicy(metadata["inputs"], metadata["hidden"]).to(
                self.device
            )
            model.load_state_dict(artifact["state_dict"])
            model.eval()
            self.models.append(model)
        self.model = self.models[0]
        self.metadata = specification

    def _features(self, board, legal):
        variants, mappings = [], []
        for reflection in (False, True):
            for k in range(4):
                transformed = np.rot90(board, k)
                mapping = (np.arange(4) - k) % 4
                if reflection:
                    transformed = transformed[:, ::-1]
                    mapping = (-mapping) % 4
                variants.append(transformed.copy())
                mappings.append(mapping.tolist())
        with torch.no_grad():
            raw = self.cnn.model(encode(np.asarray(variants), self.device)).reshape(8, 1, 4)
            aligned = torch.stack([raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)
            mask = torch.zeros((1, 4), dtype=torch.bool, device=self.device)
            mask[0, legal] = True
            masked = aligned.masked_fill(~mask[:, None], -1e9)
            choices = masked.argmax(2)
            votes = torch.stack([(choices == action).sum(1) for action in range(4)], 1)
            probabilities = torch.softmax(masked, 2).mean(1)
            expert_rows = []
            for expert in self.experts:
                expert_raw = expert.model(
                    encode(np.asarray(variants), self.device)
                ).reshape(8, 1, 4)
                expert_view = torch.stack(
                    [
                        expert_raw[i][:, mapping]
                        for i, mapping in enumerate(mappings)
                    ],
                    1,
                )
                expert_view -= expert_view.mean(2, keepdim=True)
                expert_view /= expert_view.std(2, keepdim=True) + 1e-6
                expert_rows.append(expert_view.cpu().numpy())

        aligned_np = aligned.cpu().numpy()
        aligned_np -= aligned_np.mean(2, keepdims=True)
        packed = np.asarray([pack(board)], dtype=np.uint64)
        sparse_values = np.stack(
            [model.values_batch(packed) for model in self.sparse], axis=1
        )
        sparse_values -= sparse_values.mean(2, keepdims=True)
        sparse_values /= sparse_values.std(2, keepdims=True) + 1e-6
        board_features = torch.nn.functional.one_hot(
            torch.as_tensor(board, device=self.device, dtype=torch.long).clamp(0, 15),
            16,
        ).reshape(1, -1).float()
        rest = np.concatenate(
            [
                aligned_np.reshape(1, -1),
                votes.cpu().numpy() / 8.0,
                probabilities.cpu().numpy(),
                sparse_values.reshape(1, -1),
                *[row.reshape(1, -1) for row in expert_rows],
            ],
            axis=1,
        ).astype(np.float32)
        return torch.cat(
            [board_features, torch.as_tensor(rest, device=self.device)], dim=1
        ), mask

    def act(self, board):
        legal = legal_actions(board)
        if not legal:
            raise ValueError("No legal actions")
        features, mask = self._features(np.asarray(board), legal)
        with torch.no_grad():
            logits = torch.stack(
                [model(features[:, : model.inputs]) for model in self.models]
            ).mean(0).masked_fill(~mask, -1e9)
        return int(logits.argmax(1).item())

    def close(self):
        models, self.sparse = getattr(self, "sparse", []), []
        for model in models:
            model.close()

    def __del__(self):
        self.close()
