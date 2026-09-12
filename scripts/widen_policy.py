"""Function-preserving widening of a trained policy, with optional symmetry-breaking noise."""

import argparse
import hashlib
from pathlib import Path
import torch
from bot2048.deep_rl import ConvPolicy


def widen(source, width=128, hidden=256, noise=0):
    old = source.config
    target = ConvPolicy(width=width, blocks=old["blocks"], hidden=hidden)
    if width < old["width"] or hidden < old["hidden"]:
        raise ValueError("Widening cannot shrink dimensions")
    channels = torch.arange(width) % old["width"]
    hiddens = torch.arange(hidden) % old["hidden"]
    cc = torch.bincount(channels, minlength=old["width"])[channels].float()
    hc = torch.bincount(hiddens, minlength=old["hidden"])[hiddens].float()
    src = source.state_dict()
    out = target.state_dict()
    out["net.0.weight"] = src["net.0.weight"][channels].clone()
    out["net.0.bias"] = src["net.0.bias"][channels].clone()
    for block in range(old["blocks"]):
        for layer in [0, 2]:
            key = f"net.{2+block}.net.{layer}"
            out[key + ".weight"] = (
                src[key + ".weight"][channels][:, channels] / cc[None, :, None, None]
            )
            out[key + ".bias"] = src[key + ".bias"][channels].clone()
    fc = old["blocks"] + 3
    last = fc + 2
    flat = (channels[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
    fc_counts = cc[:, None].expand(-1, 16).reshape(-1)
    out[f"net.{fc}.weight"] = (
        src[f"net.{fc}.weight"][hiddens][:, flat] / fc_counts[None, :]
    )
    out[f"net.{fc}.bias"] = src[f"net.{fc}.bias"][hiddens].clone()
    out[f"net.{last}.weight"] = src[f"net.{last}.weight"][:, hiddens] / hc[None, :]
    out[f"net.{last}.bias"] = src[f"net.{last}.bias"].clone()
    if noise:
        for name, tensor in out.items():
            if name.endswith("weight"):
                out[name] = tensor + torch.randn_like(tensor) * tensor.std() * noise
    target.load_state_dict(out)
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/rl/conv_extended.pt")
    p.add_argument("--output", default="models/rl/conv_wide_initial.pt")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--noise", type=float, default=0.001)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    data = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    source = ConvPolicy(**data["metadata"].get("network_config", {}))
    source.load_state_dict(data["state_dict"])
    exact = widen(source, a.width, a.hidden, 0)
    x = torch.randn(16, 16, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(source(x), exact(x), rtol=1e-4, atol=1e-4)
    model = widen(source, a.width, a.hidden, a.noise)
    metadata = {
        **data["metadata"],
        "network_config": model.config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "widening": vars(a),
        "source_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, a.output)
    print(
        f"Verified exact widening, then saved {metadata['parameters']:,} parameters with noise {a.noise}"
    )


if __name__ == "__main__":
    main()
