"""Add identity-initialized residual blocks while preserving policy outputs."""

import argparse
import hashlib
from pathlib import Path

import torch

from bot2048.deep_rl import ConvPolicy


def deepen(source, blocks=5):
    old = source.config
    if blocks < old["blocks"]:
        raise ValueError("Deepening cannot remove residual blocks")
    target = ConvPolicy(width=old["width"], blocks=blocks, hidden=old["hidden"])
    src = source.state_dict()
    out = target.state_dict()
    out["net.0.weight"] = src["net.0.weight"].clone()
    out["net.0.bias"] = src["net.0.bias"].clone()
    for block in range(old["blocks"]):
        for layer in (0, 2):
            prefix = f"net.{2 + block}.net.{layer}"
            out[prefix + ".weight"] = src[prefix + ".weight"].clone()
            out[prefix + ".bias"] = src[prefix + ".bias"].clone()
    for block in range(old["blocks"], blocks):
        first = f"net.{2 + block}.net.0"
        second = f"net.{2 + block}.net.2"
        # A zero second convolution makes the residual block exactly the identity.
        torch.nn.init.kaiming_normal_(out[first + ".weight"], nonlinearity="relu")
        out[first + ".bias"].zero_()
        out[second + ".weight"].zero_()
        out[second + ".bias"].zero_()
    old_fc = old["blocks"] + 3
    new_fc = blocks + 3
    old_output = old_fc + 2
    new_output = new_fc + 2
    for suffix in ("weight", "bias"):
        out[f"net.{new_fc}.{suffix}"] = src[f"net.{old_fc}.{suffix}"].clone()
        out[f"net.{new_output}.{suffix}"] = src[f"net.{old_output}.{suffix}"].clone()
    target.load_state_dict(out)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--blocks", type=int, default=5)
    a = p.parse_args()
    torch.set_num_threads(2)
    data = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    source = ConvPolicy(**data["metadata"].get("network_config", {}))
    source.load_state_dict(data["state_dict"])
    target = deepen(source, a.blocks)
    with torch.no_grad():
        sample = torch.randn(32, 16, 4, 4)
        torch.testing.assert_close(source(sample), target(sample), rtol=0, atol=0)
    metadata = {
        **data["metadata"],
        "network_config": target.config,
        "parameters": sum(p.numel() for p in target.parameters()),
        "deepening": vars(a),
        "source_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "function_preserved": True,
    }
    torch.save({"state_dict": target.state_dict(), "metadata": metadata}, a.output)
    print(
        f"Saved exact {a.blocks}-block expansion with {metadata['parameters']:,} parameters"
    )


if __name__ == "__main__":
    main()
