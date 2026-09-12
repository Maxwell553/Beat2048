"""Compact convolutional action policy. Inference never evaluates successor boards."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from .env import legal_actions


class Residual(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class ConvPolicy(nn.Module):
    def __init__(self, width=64, blocks=3, hidden=128, adapter_hidden=0):
        super().__init__()
        self.config = {
            "width": width,
            "blocks": blocks,
            "hidden": hidden,
            "adapter_hidden": adapter_hidden,
        }
        self.net = nn.Sequential(
            nn.Conv2d(16, width, 3, padding=1),
            nn.ReLU(),
            *[Residual(width) for _ in range(blocks)],
            nn.Flatten(),
            nn.Linear(16 * width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)
        )
        self.adapter = None
        if adapter_hidden:
            self.adapter = nn.Sequential(
                nn.Linear(hidden, adapter_hidden),
                nn.ReLU(),
                nn.Linear(adapter_hidden, 4),
            )
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        if self.adapter is None:
            return self.net(x)
        features = self.net[:-1](x)
        return self.net[-1](features) + self.adapter(features)


class AfterstateValueNet(nn.Module):
    """Scalar neural value function shared by all four deterministic moves."""

    def __init__(self, width=96, blocks=5, hidden=256):
        super().__init__()
        self.config = {
            "kind": "afterstate_value",
            "width": width,
            "blocks": blocks,
            "hidden": hidden,
        }
        self.net = nn.Sequential(
            nn.Conv2d(16, width, 3, padding=1),
            nn.ReLU(),
            *[Residual(width) for _ in range(blocks)],
            nn.Flatten(),
            nn.Linear(16 * width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class AfterstateValueNet(nn.Module):
    """Scalar neural value model for a board after a deterministic move."""

    def __init__(self, width=96, blocks=5, hidden=256):
        super().__init__()
        self.config = {
            "kind": "afterstate_value",
            "width": width,
            "blocks": blocks,
            "hidden": hidden,
        }
        self.net = nn.Sequential(
            nn.Conv2d(16, width, 3, padding=1),
            nn.ReLU(),
            *[Residual(width) for _ in range(blocks)],
            nn.Flatten(),
            nn.Linear(16 * width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class EquivariantConvPolicy(ConvPolicy):
    """D4-equivariant current-board policy with differentiable soft voting."""

    def __init__(
        self, width=64, blocks=3, hidden=128, adapter_hidden=0, vote_temperature=0.05
    ):
        super().__init__(width, blocks, hidden, adapter_hidden)
        self.vote_temperature = vote_temperature
        self.config = {
            "kind": "equivariant_conv",
            "width": width,
            "blocks": blocks,
            "hidden": hidden,
            "adapter_hidden": adapter_hidden,
            "vote_temperature": vote_temperature,
        }

    def _raw(self, x):
        if self.adapter is None:
            return self.net(x)
        features = self.net[:-1](x)
        return self.net[-1](features) + self.adapter(features)

    def forward(self, x):
        variants, mappings = [], []
        for reflection in (False, True):
            for k in range(4):
                variant = torch.rot90(x, k, dims=(-2, -1))
                mapping = (np.arange(4) - k) % 4
                if reflection:
                    variant = variant.flip(-1)
                    mapping = (-mapping) % 4
                variants.append(variant)
                mappings.append(mapping.tolist())
        raw = self._raw(torch.cat(variants)).reshape(8, len(x), 4)
        aligned = torch.stack(
            [raw[i][:, mapping] for i, mapping in enumerate(mappings)]
        )
        return torch.log_softmax(
            aligned / self.vote_temperature, dim=2
        ).logsumexp(0)


class SymmetryHead(nn.Module):
    """Learn how to combine eight aligned policy views of one current board."""

    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(8 * 4 + 16 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 4),
        )

    def forward(self, aligned_logits, encoded_board):
        features = torch.cat(
            [aligned_logits.flatten(1), encoded_board.flatten(1)], dim=1
        )
        return self.net(features)


class ResidualSymmetryHead(nn.Module):
    """A bounded correction to the policy's exact eight-view vote scores.

    Its final layer starts at zero, so an untrained head leaves every deployed
    decision unchanged. This permits rare corrections without relearning the
    strong base policy first.
    """

    def __init__(self, hidden=256, scale=4.0):
        super().__init__()
        self.hidden = hidden
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(8 * 4 + 16 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def vote_scores(aligned_logits, legal=None):
        voting_logits = aligned_logits
        if legal is not None:
            voting_logits = voting_logits.masked_fill(~legal[:, None, :], -1e9)
        choices = voting_logits.argmax(2)
        votes = torch.stack(
            [(choices == action).sum(1) for action in range(4)], dim=1
        ).float()
        return votes + 0.99 * torch.softmax(voting_logits, dim=2).mean(1)

    def forward(self, aligned_logits, encoded_board, legal=None):
        features = torch.cat(
            [aligned_logits.flatten(1), encoded_board.flatten(1)], dim=1
        )
        correction = self.scale * torch.tanh(self.net(features))
        return self.vote_scores(aligned_logits, legal) + correction


class GatedSymmetryHead(nn.Module):
    """Selectively replace a neural vote when a learned risk gate fires."""

    def __init__(self, hidden=256, threshold=0.5):
        super().__init__()
        self.hidden = hidden
        self.threshold = threshold
        self.features = nn.Sequential(
            nn.Linear(8 * 4 + 16 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.gate = nn.Linear(hidden // 2, 1)
        self.actions = nn.Linear(hidden // 2, 4)

    def raw(self, aligned_logits, encoded_board):
        features = torch.cat(
            [aligned_logits.flatten(1), encoded_board.flatten(1)], dim=1
        )
        hidden = self.features(features)
        return self.gate(hidden).squeeze(1), self.actions(hidden)

    def forward(self, aligned_logits, encoded_board, legal=None):
        base = ResidualSymmetryHead.vote_scores(aligned_logits, legal)
        gate, actions = self.raw(aligned_logits, encoded_board)
        if legal is not None:
            actions = actions.masked_fill(~legal, -1e9)
        return torch.where((torch.sigmoid(gate) >= self.threshold)[:, None], actions, base)


class FusionPolicy(nn.Module):
    """Fuse board, convolutional-view, and sparse-policy neural features."""

    def __init__(self, inputs=308, hidden=512):
        super().__init__()
        self.inputs = inputs
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 4),
        )

    def forward(self, features):
        return self.net(features)


class GatedFusionCorrection(nn.Module):
    """Conservatively select an outcome-trained action over a frozen base policy."""

    def __init__(self, inputs=308, hidden=64):
        super().__init__()
        self.inputs, self.hidden = inputs, hidden
        self.features = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.gate = nn.Linear(hidden, 1)
        self.actions = nn.Linear(hidden, 4)

    def forward(self, features):
        hidden = self.features(features)
        return self.gate(hidden).squeeze(1), self.actions(hidden)


class CommitteeSelector(nn.Module):
    """Choose between two frozen neural committees from current-board features."""

    def __init__(self, inputs=340, hidden=128):
        super().__init__()
        self.inputs, self.hidden = inputs, hidden
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(1)


class DenseResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class DensePolicy(nn.Module):
    """Residual MLP policy over the complete one-hot current board."""

    def __init__(self, width=512, blocks=3):
        super().__init__()
        self.config = {"kind": "dense", "width": width, "blocks": blocks}
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, width),
            nn.ReLU(),
            *[DenseResidual(width) for _ in range(blocks)],
            nn.Linear(width, 4),
        )

    def forward(self, x):
        return self.net(x)


class DirectionalPolicy(nn.Module):
    """Policy with row- and column-spanning filters tailored to 2048 moves."""

    def __init__(self, width=128, hidden=512, blocks=2):
        super().__init__()
        self.config = {
            "kind": "directional",
            "width": width,
            "hidden": hidden,
            "blocks": blocks,
        }
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(16, width, kernel)
                for size in (2, 3, 4)
                for kernel in ((1, size), (size, 1))
            ]
        )
        branch_features = width * 48
        self.input = nn.Sequential(nn.Linear(branch_features, hidden), nn.ReLU())
        self.residuals = nn.Sequential(
            *[DenseResidual(hidden) for _ in range(blocks)]
        )
        self.output = nn.Linear(hidden, 4)

    def forward(self, x):
        features = torch.cat(
            [torch.relu(branch(x)).flatten(1) for branch in self.branches], dim=1
        )
        return self.output(self.residuals(self.input(features)))


class TransformerPolicy(nn.Module):
    """Compact board-token transformer for global current-board relationships."""

    def __init__(self, width=64, hidden=256, blocks=4, heads=8):
        super().__init__()
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        self.config = {
            "kind": "transformer", "width": width, "hidden": hidden,
            "blocks": blocks, "heads": heads,
        }
        self.embedding = nn.Linear(16, width, bias=False)
        self.position = nn.Parameter(torch.zeros(1, 16, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=hidden, dropout=0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, blocks)
        self.output = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(16 * width), nn.Linear(16 * width, hidden),
            nn.GELU(), nn.Linear(hidden, 4),
        )

    def forward(self, x):
        tokens = x.permute(0, 2, 3, 1).reshape(len(x), 16, 16)
        return self.output(self.encoder(self.embedding(tokens) + self.position))


class LineEmbeddingPolicy(nn.Module):
    """Current-board policy with learned lookup embeddings for complete lines.

    A 2048 move acts independently on each row or column before the random tile
    appears.  Encoding each complete four-cell line gives the network a compact
    learned representation of merge structure without constructing successor
    boards or hard-coding the slide operation at inference time.
    """

    def __init__(self, embedding=64, hidden=512, blocks=3):
        super().__init__()
        self.config = {
            "kind": "line_embedding",
            "embedding": embedding,
            "hidden": hidden,
            "blocks": blocks,
        }
        self.lines = nn.Embedding(16 ** 4, embedding)
        inputs = 8 * embedding + 16 * 4 * 4
        self.input = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU())
        self.residuals = nn.Sequential(
            *[DenseResidual(hidden) for _ in range(blocks)]
        )
        self.output = nn.Linear(hidden, 4)
        self.register_buffer(
            "line_powers", torch.tensor([1, 16, 256, 4096], dtype=torch.long)
        )

    def forward(self, x):
        ranks = x.argmax(1)
        rows = (ranks * self.line_powers).sum(2)
        columns = (ranks.transpose(1, 2) * self.line_powers).sum(2)
        line_features = self.lines(torch.cat([rows, columns], dim=1)).flatten(1)
        features = torch.cat([line_features, x.flatten(1)], dim=1)
        return self.output(self.residuals(self.input(features)))


class CanonicalLinePolicy(nn.Module):
    """Rotation-equivariant policy with one shared learned directional scorer."""

    def __init__(self, embedding=64, hidden=256, blocks=2):
        super().__init__()
        self.config = {
            "kind": "canonical_line",
            "embedding": embedding,
            "hidden": hidden,
            "blocks": blocks,
        }
        self.lines = nn.Embedding(16 ** 4, embedding)
        self.input = nn.Sequential(
            nn.Linear(4 * embedding + 16 * 4 * 4, hidden), nn.ReLU()
        )
        self.residuals = nn.Sequential(
            *[DenseResidual(hidden) for _ in range(blocks)]
        )
        self.output = nn.Linear(hidden, 1)
        self.register_buffer(
            "line_powers", torch.tensor([1, 16, 256, 4096], dtype=torch.long)
        )

    def _score_left(self, x):
        ranks = x.argmax(1)
        rows = (ranks * self.line_powers).sum(2)
        line_features = self.lines(rows).flatten(1)
        features = torch.cat([line_features, x.flatten(1)], dim=1)
        return self.output(self.residuals(self.input(features))).squeeze(1)

    def forward(self, x):
        # Actions are up, right, down, left. Rotating each current board by
        # 1, 2, 3, 0 quarter turns maps that action to canonical left.
        views = torch.cat([torch.rot90(x, (action + 1) % 4, (-2, -1))
                           for action in range(4)])
        return self._score_left(views).reshape(4, len(x)).transpose(0, 1)


def policy_from_config(config):
    config = dict(config)
    kind = config.pop("kind", "conv")
    if kind == "dense":
        return DensePolicy(**config)
    if kind == "directional":
        return DirectionalPolicy(**config)
    if kind == "equivariant_conv":
        return EquivariantConvPolicy(**config)
    if kind == "transformer":
        return TransformerPolicy(**config)
    if kind == "line_embedding":
        return LineEmbeddingPolicy(**config)
    if kind == "canonical_line":
        return CanonicalLinePolicy(**config)
    if kind == "afterstate_value":
        return AfterstateValueNet(**config)
    if kind == "afterstate_value":
        return AfterstateValueNet(**config)
    if kind != "conv":
        raise ValueError(f"unknown policy kind: {kind}")
    return ConvPolicy(**config)


def encode(boards, device="cpu"):
    ranks = torch.as_tensor(np.asarray(boards, dtype=np.int64), device=device)
    return nn.functional.one_hot(ranks.clamp(0, 15), 16).permute(0, 3, 1, 2).float()


class ConvAgent:
    def __init__(
        self,
        checkpoint,
        device="cpu",
        ensemble=False,
        voting=False,
        extra_checkpoints=None,
    ):
        self.voting = voting
        self.ensemble = ensemble
        self.device = device
        paths = [checkpoint, *(extra_checkpoints or [])]
        artifacts = [
            torch.load(path, map_location=device, weights_only=True) for path in paths
        ]
        self.models = []
        for artifact in artifacts:
            model = policy_from_config(
                artifact["metadata"].get("network_config", {})
            ).to(device)
            model.load_state_dict(artifact["state_dict"])
            model.eval()
            self.models.append(model)
        self.model = self.models[0]
        self.head = None
        self.head_min_confidence = 0.0
        if "head_state_dict" in artifacts[0]:
            if len(artifacts) > 1:
                raise ValueError("learned symmetry heads cannot form a committee")
            head_config = artifacts[0]["metadata"]["symmetry_head"]
            if head_config.get("kind") == "residual_vote":
                self.head = ResidualSymmetryHead(
                    head_config["hidden"], head_config.get("scale", 4.0)
                ).to(device)
            elif head_config.get("kind") == "gated_vote":
                self.head = GatedSymmetryHead(
                    head_config["hidden"], head_config.get("threshold", 0.5)
                ).to(device)
            else:
                self.head = SymmetryHead(head_config["hidden"]).to(device)
            self.head.load_state_dict(artifacts[0]["head_state_dict"])
            self.head.eval()
            self.ensemble = True
            self.head_min_confidence = artifacts[0]["metadata"]["symmetry_head"].get(
                "minimum_confidence", 0.0
            )
        self.metadata = artifacts[0]["metadata"]
        if len(paths) > 1:
            self.metadata = dict(self.metadata)
            self.metadata["neural_committee"] = [str(path) for path in paths]

    def act(self, board):
        legal = legal_actions(board)
        if not legal:
            raise ValueError("No legal actions")
        with torch.no_grad():
            if self.ensemble:
                variants = []
                mappings = []
                for reflection in (False, True):
                    for k in range(4):
                        transformed = np.rot90(board, k)
                        mapping = (np.arange(4) - k) % 4
                        if reflection:
                            transformed = transformed[:, ::-1]
                            mapping = (-mapping) % 4
                        variants.append(transformed.copy())
                        mappings.append(mapping)
                encoded = encode(np.asarray(variants), self.device)
                predictions = [model(encoded).cpu().numpy() for model in self.models]
                aligned = np.concatenate(
                    [
                        np.asarray([raw[i, m] for i, m in enumerate(mappings)])
                        for raw in predictions
                    ]
                )
                if self.head is not None:
                    aligned_tensor = torch.as_tensor(
                        aligned[None], device=self.device, dtype=torch.float32
                    )
                    current = encode(np.asarray(board)[None], self.device)
                    if isinstance(self.head, (ResidualSymmetryHead, GatedSymmetryHead)):
                        legal_mask = torch.zeros(
                            (1, 4), device=self.device, dtype=torch.bool
                        )
                        legal_mask[0, legal] = True
                        head_logits = self.head(
                            aligned_tensor, current, legal_mask
                        )[0]
                    else:
                        head_logits = self.head(aligned_tensor, current)[0]
                    q = head_logits.cpu().numpy()
                    confidence = torch.softmax(head_logits[legal], dim=0).max().item()
                    if confidence < self.head_min_confidence:
                        valid = aligned[:, legal]
                        probabilities = torch.softmax(
                            torch.tensor(valid), dim=1
                        ).numpy()
                        votes = np.bincount(valid.argmax(1), minlength=len(legal))
                        q = np.full(4, -1e30, dtype=float)
                        q[legal] = votes + 0.99 * probabilities.mean(0)
                elif self.voting:
                    valid = aligned[:, legal]
                    probabilities = torch.softmax(torch.tensor(valid), dim=1).numpy()
                    votes = np.bincount(valid.argmax(1), minlength=len(legal))
                    q = np.full(4, -1e30, dtype=float)
                    q[legal] = votes + 0.99 * probabilities.mean(0)
                else:
                    q = aligned.mean(0)
            else:
                encoded = encode(np.asarray(board)[None], self.device)
                q = np.mean(
                    [model(encoded)[0].cpu().numpy() for model in self.models], axis=0
                )
        return int(legal[np.argmax(q[legal])])
