"""Build a reviewable release without multi-gigabyte rejected experiments."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "2048-bot.zip"
    files = []
    for directory in ["bot2048", "native", "scripts", "tests"]:
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".h"}:
                files.append(path)
    for name in [
        "README.md",
        "RL_PROGRESS.md",
        "requirements.txt",
        "requirements-lock.txt",
        "THIRD_PARTY_NOTICES.md",
        "models/rl/final_value_policy.json",
        "models/rl/value_teacher_500k.ptbin.gz",
        "models/rl/value_teacher_510k.ptbin.gz",
        "models/rl/value_teacher_continued_100k.ptbin.gz",
        "models/rl/conv_initial.pt",
        "results/rl/final_rl_value_ensemble_700000.json",
        "results/rl/teacher_ensemble_base_1111.json",
        "results/artifact_checks.json",
        "animations/random_agent.gif",
        "animations/random_agent.json",
        "animations/random_agent_final.png",
        "animations/rl_initial_neural.gif",
        "animations/rl_initial_neural.json",
        "animations/rl_initial_neural_final.png",
        "animations/final_rl_bot.gif",
        "animations/final_rl_bot.json",
        "animations/final_rl_bot_final.png",
        "reel_01/REEL_SCRIPT.md",
    ]:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"required release artifact is missing: {name}")
        files.append(path)
    files = sorted(set(files))
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("2048-bot") / path.relative_to(root))
    with ZipFile(output) as archive:
        assert archive.testzip() is None
    print(
        f"Saved {output.name}: {len(files)} files, "
        f"{output.stat().st_size:,} bytes; ZIP CRCs verified"
    )


if __name__ == "__main__":
    main()
