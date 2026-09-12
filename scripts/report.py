"""Training plots and an artifact manifest computed from the finished run."""

import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    history = json.loads(Path("results/training.json").read_text())["history"]
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    epochs = [x["epoch"] for x in history]
    for key, label, color in [
        ("train_loss", "Training", "#64dfcb"),
        ("validation_loss", "Validation", "#ffbd69"),
    ]:
        axes[0].plot(epochs, [x[key] for x in history], label=label, color=color)
    for key, label, color in [
        ("train_accuracy", "Training", "#64dfcb"),
        ("validation_accuracy", "Validation", "#ffbd69"),
    ]:
        axes[1].plot(epochs, [100 * x[key] for x in history], label=label, color=color)
    axes[0].set(
        ylabel="Cross-entropy loss", xlabel="Epoch", title="Policy distillation: loss"
    )
    axes[1].set(
        ylabel="Teacher action agreement (%)",
        xlabel="Epoch",
        title="Policy distillation: agreement",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
    fig.savefig("results/training_curve.png", dpi=170)
    plt.close(fig)
    evaluation = Path("results/evaluation.json")
    if evaluation.exists():
        evaluation_data = json.loads(evaluation.read_text())
        summary = evaluation_data["summary"]
        first_seed = evaluation_data["config"]["seed"]
        last_seed = first_seed + evaluation_data["config"]["games"] - 1
        names = list(summary)
        values = [100 * summary[m]["win_rate"] for m in names]
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        bars = ax.bar(
            names,
            values,
            color=["#778899", "#9489b5", "#70b4eb", "#ffbd69", "#64dfcb"][: len(names)],
        )
        for bar, m in zip(bars, names):
            s = summary[m]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{s['wins']}/{s['games']}",
                ha="center",
            )
        ax.set(
            ylim=(0, 112),
            ylabel="Games reaching 2048 (%)",
            title=f"Independent benchmark — seeds {first_seed}–{last_seed}",
        )
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig("results/win_rates.png", dpi=170)
        plt.close(fig)
    entries = []
    for directory in ["models", "animations", "data", "results"]:
        for p in sorted(Path(directory).glob("*")):
            if p.is_file() and p.name != "manifest.json":
                entries.append(
                    {
                        "path": str(p),
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                )
    Path("results/manifest.json").write_text(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
