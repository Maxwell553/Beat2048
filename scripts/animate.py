"""Render honest recordings from saved trajectories; no generated gameplay images."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.evaluate import play

COLORS = {
    0: "#263344",
    1: "#eee4da",
    2: "#ede0c8",
    3: "#f2b179",
    4: "#f59563",
    5: "#f67c5f",
    6: "#f65e3b",
    7: "#edcf72",
    8: "#edcc61",
    9: "#edc850",
    10: "#edc53f",
    11: "#edc22e",
}


def font(size):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def frame(state, title, subtitle, finished=False, won=False, seed=42):
    im = Image.new("RGB", (640, 800), "#111b2a")
    d = ImageDraw.Draw(im)
    d.text((34, 25), "2048 / LEARNING TO PLAY", font=font(19), fill="#70d6ce")
    d.text((34, 61), title, font=font(32), fill="#f4f6f9")
    d.text((34, 109), subtitle, font=font(15), fill="#aab8cb")
    d.text((34, 150), f"SCORE  {state['score']:,}", font=font(22), fill="white")
    d.text((388, 150), f"MOVE  {state['move']:,}", font=font(22), fill="white")
    d.rounded_rectangle((26, 199, 614, 787), radius=18, fill="#39475b")
    board = np.array(state["board"])
    for r in range(4):
        for c in range(4):
            rank = int(board[r, c])
            x = 38 + c * 144
            y = 211 + r * 144
            d.rounded_rectangle(
                (x, y, x + 132, y + 132), radius=10, fill=COLORS.get(rank, "#7569d6")
            )
            if rank:
                text = str(2**rank)
                f = font(46 if len(text) < 4 else 36)
                box = d.textbbox((0, 0), text, font=f)
                d.text(
                    (
                        x + (132 - (box[2] - box[0])) / 2,
                        y + (132 - (box[3] - box[1])) / 2 - box[1],
                    ),
                    text,
                    font=f,
                    fill="#5b524a" if rank < 3 else "#fffaf3",
                )
    if finished:
        d.rounded_rectangle(
            (91, 425, 549, 559), radius=18, fill="#111b2a", outline="#70d6ce", width=3
        )
        label = "2048 REACHED" if won else "GAME OVER"
        f = font(32)
        box = d.textbbox((0, 0), label, font=f)
        d.text(
            ((640 - box[2]) / 2, 453),
            label,
            font=f,
            fill="#70d6ce" if won else "#ff9b8c",
        )
        d.text(
            (170, 505),
            f"Recorded simulation • seed {seed}",
            font=font(17),
            fill="#aab8cb",
        )
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default="models/final_policy.pt")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--cutoff", type=float, default=0.0001)
    p.add_argument("--neural-weight", type=float, default=0.02)
    a = p.parse_args()
    Path("animations").mkdir(exist_ok=True)
    configs = [
        (
            "random",
            None,
            "random_agent",
            "Random agent",
            "Uniform legal actions • untrained",
        ),
        (
            "neural",
            "models/initial_policy.pt",
            "initial_neural",
            "Initial neural network",
            "Random weights • legal-action argmax",
        ),
        (
            "hybrid",
            a.checkpoint,
            "trained_bot",
            "Trained bot + search",
            "Learned policy + expectimax • accelerated replay",
        ),
    ]
    for mode, checkpoint, name, title, subtitle in configs:
        row = play((mode, checkpoint, a.seed, a.depth, a.cutoff, a.neural_weight, True))
        Path(f"animations/{name}.json").write_text(
            json.dumps({"config": vars(a), **row}, indent=2)
        )
        trajectory = row["trajectory"]
        indices = np.unique(
            np.linspace(0, len(trajectory) - 1, min(150, len(trajectory)), dtype=int)
        )
        frames = [frame(trajectory[i], title, subtitle) for i in indices]
        final = frame(trajectory[-1], title, subtitle, True, row["won"], a.seed)
        frames.append(final)
        frames[0].save(
            f"animations/{name}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=[90] * (len(frames) - 1) + [2200],
            loop=0,
            optimize=True,
        )
        final.save(f"animations/{name}_final.png")
        print(
            json.dumps({k: v for k, v in row.items() if k != "trajectory"}), flush=True
        )


if __name__ == "__main__":
    main()
