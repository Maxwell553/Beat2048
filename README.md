# Search-free reinforcement-learned 2048 bot

This project contains a reproducible standard 2048 environment, reinforcement-learning code, saved initial and final agents, replayable animations, and a frozen bot that won **100/100 previously untouched certification games**.

The default bot does **not** use expectimax. It does not enumerate possible random tile spawns, attach probabilities to future boards, recurse through a tree, or use a handcrafted board score. It applies each legal move once, then asks a reinforcement-learned sparse neural value ensemble to score those deterministic afterstates. The highest learned value wins.

```bash
python3 -m bot2048 --seed 42
python3 -m bot2048 --seed 42 --quiet
```

The frozen manifest is [`models/rl/final_value_policy.json`](models/rl/final_value_policy.json). It names every checkpoint, weight, SHA-256 digest, inference rule, and certification result. The loader refuses altered checkpoint bytes.

## Results

All reported games use ordinary two-tile starts and normal random spawns.

| Agent | Games | Wins | Mean moves | Search |
| --- | ---: | ---: | ---: | --- |
| Final RL value ensemble, certification seeds 700000–700099 | **100** | **100** | 1,181.54 | No |
| Same architecture, broad fresh evaluation | 10,000 | 9,869 | 1,170.64 | No |
| Best current-board-only dense neural committee | 100 | 79 | — | No |
| Random-weight neural baseline | 100 | 0 | — | No |

The complete certification record is [`results/rl/final_rl_value_ensemble_700000.json`](results/rl/final_rl_value_ensemble_700000.json). The 100 games were reserved throughout development and evaluated once after the ensemble and weights were frozen. A 100/100 sample is empirical certification, not a proof covering every possible infinite sequence of random spawns. Its Wilson 95% interval is 96.30%–100%.

The 10,000-game measurement provides useful context: the same architecture won 98.69% of that independent range. It prevents the perfect 100-game sample from being mistaken for a mathematical guarantee.

## Realistic game rules

A new game begins with **exactly two occupied cells**, not four. Each initial tile and each later tile is a 2 with 90% probability or a 4 with 10% probability, placed uniformly in an empty cell. A move compresses toward an edge, merges equal neighbors once, scores the merged value, compresses again, and spawns one tile. A game counts as a win when it first reaches 2048.

The full alternating board below cannot be an initial position:

```text
2 4 2 4
4 2 4 2
2 4 2 4
4 2 4 2
```

It appears only as an artificial terminal-state unit test. Tests need edge cases even when reset cannot produce them. Every training and evaluation game uses the realistic two-tile start.

`bot2048/env.py` is the independent reference environment. The native training environment in `native/rl_engine.h` was checked against it for moves, merge rewards, legal-action masks, and spawn behavior.

## What the neural network does

The final learner is an n-tuple neural value function. Each six-cell pattern selects one learned scalar weight, and the active pattern weights are summed. This is a sparse feed-forward neural architecture: lookup units form the input features, their learned weights form a linear value layer, and only units matching the board are active. Rotated and reflected pattern copies share the same value objective.

For one decision:

1. The environment computes the deterministic result of each legal move.
2. Every saved value network scores each resulting board once.
3. The three learned predictions are combined with frozen weights 1:1:2.
4. The bot selects the legal action with the largest combined value.

This is one-ply neural action scoring, not expectimax. There are no random-spawn children and no second move layer. Runtime grows linearly with the four actions rather than exponentially with a search depth.

The network is small in active computation: only pattern units selected by the board are read for a value prediction. The checkpoint is comparatively large because the sparse table reserves a weight for every hashed six-tile pattern. Three 256 MiB snapshots reduce independent errors. This memory/performance tradeoff is explicit rather than presenting the model as a compact dense CNN.

The learned RL value is the main driver of performance. The deterministic move operation supplies the game rules and contains no preference among legal moves. Removing the learned value leaves no strategy.

## What expectimax was doing

Expectimax belongs to the original comparison experiment. It expanded future player moves and random spawns, then backed up expected heuristic values. That planner once produced a 100/100 result while the accompanying dense neural policy was weak, so it did not satisfy the neural-performance goal.

The default CLI mode is now `value`. Search code loads only for an explicit legacy command:

```bash
python3 -m bot2048 --mode search --depth 4 --seed 42
```

The certified path never calls that planner. The final manifest records `"search": false`, and the result files repeat the exact inference rule.

## Reinforcement-learning method

Training uses afterstate TD(λ). An afterstate is the board immediately after a chosen slide/merge and before the random tile appears. This representation separates the player's action from the environment's random transition.

For each episode:

1. Start from two normally spawned tiles.
2. Score every legal afterstate using the current learned value plus immediate merge reward.
3. Take the greedy action and sample the real random spawn.
4. Save the afterstate, predicted value, and reward.
5. Walk backward through the trajectory and update active tuple weights with a TD(λ) target.

The value function learns only from merge rewards and its own played trajectories. It receives no expectimax labels and no handcrafted monotonicity, corner, smoothness, or empty-cell features.

Several snapshots from continuation points form the final ensemble. Averaging helps because snapshots make partially different rare late-game errors. Ensemble weights were frozen on development data before the certification range was opened.

```bash
c++ -O3 -std=c++17 -march=native native/td_value.cpp -o native/td_value
./native/td_value train 500000 models/rl/new_value.bin 50000000 .2 .5
./native/td_value resume 100000 models/rl/new_value.bin 60000000 .05 .5
```

Checkpoint writes are atomic. Episode seeds derive deterministically from the supplied starting seed.

## Dense-network experiments

The source tree also preserves the conventional dense neural pipeline used during development. Its strongest current-board-only candidate combined a 2,019,972-parameter residual CNN, eight rotation/reflection views, three sparse direct-action networks, a whole-line embedding expert, and two fusion MLP heads.

That policy performs current-board forward passes and never constructs afterstates. It improved from 1/100 for early distillation, to 71/100 for the deeper CNN, to 74/100 for CNN/sparse fusion, and finally to **79/100** for the heterogeneous committee.

The dense approach was not used for the final claim because it did not meet the 100/100 gate. Its large intermediate datasets and rejected checkpoints were removed from the GitHub-ready repository; `RL_PROGRESS.md` retains their results and development history. High training-label accuracy is not the same as closed-loop reliability.

## Problems encountered

**Sparse reward.** Random policies almost never reach 2048, so early PPO runs had little useful success signal. Afterstate TD learning was much stronger because merges supply frequent rewards and the value target stays close to the decision boundary.

**Distribution shift.** A distilled CNN may copy most teacher actions but one mistake changes every later state. DAgger-style collection added millions of learner-visited positions and replayed windows before failures. It improved the dense policy without closing the gap.

**Validation did not predict gameplay perfectly.** Several checkpoints improved teacher agreement while losing more games. Promotions therefore required completed games on fixed development seeds. The final range stayed sealed until the architecture and weights were frozen.

**Symmetry disagreement.** Rotations and reflections should be strategically equivalent, yet ordinary CNN predictions differed across views. Neural voting helped. A line-embedding expert was weak by itself but complementary in the fusion committee.

**Outcome-label variance.** Rollouts from failure states suggested better first actions, but four stochastic continuations per action were too noisy for a correction gate to generalize. Those heads were rejected.

**Architecture dead ends.** Wider sparse direct-action networks, a board-token transformer, an explicitly equivariant CNN, learned selectors, PPO residual heads, and regret-weighted fusion refinements all failed paired gameplay checks.

**Dense afterstate approximation.** A 1.1-million-parameter scalar residual CNN factored exact move mechanics from learned value. It reached 66.5% held-out teacher-action agreement but only 3/64 gameplay wins, so it was rejected.

**Teacher continuation regression.** A wider tuple learner improved through 350k games, then regressed after continuing to 500k. Fresh paired evaluation caught the regression. Later training is not assumed to be better.

`RL_PROGRESS.md` preserves the chronology and negative results.

## Saved artifacts

- [`models/rl/final_value_policy.json`](models/rl/final_value_policy.json): final manifest and hashes.
- `models/rl/value_teacher_500k.ptbin.gz`, `value_teacher_510k.ptbin.gz`, and `value_teacher_continued_100k.ptbin.gz`: exact compressed RL neural value snapshots. Each is below GitHub's 100 MB file limit and expands into a verified system cache on first use.
- [`results/rl/final_rl_value_ensemble_700000.json`](results/rl/final_rl_value_ensemble_700000.json): untouched 100/100 certification.
- [`animations/final_rl_bot.gif`](animations/final_rl_bot.gif): accelerated final-agent game, seed 42.
- [`animations/final_rl_bot.json`](animations/final_rl_bot.json): complete replay-verified trace.
- [`animations/random_agent.gif`](animations/random_agent.gif): uniform-random legal agent.
- [`animations/rl_initial_neural.gif`](animations/rl_initial_neural.gif): saved random-weight dense neural agent.
- `models/rl/conv_initial.pt`: saved random-weight dense checkpoint.
- `bot2048/`, `native/`, `scripts/`, and `tests/`: environment, models, training, evaluation, and checks.

The GIFs are rendered from saved trajectories rather than generated illustrations. Their JSON traces replay move by move in the independent Python environment.

## Repository storage

The production checkpoints are stored as deterministic gzip files so every tracked file remains below GitHub's 100 MB limit. On first use, the loader verifies each compressed SHA-256 digest, expands the three checkpoints into the operating system's temporary cache, and verifies their original digests before native inference. The cache needs about 768 MiB and can be regenerated at any time.

Training corpora, rejected checkpoints, intermediate evaluations, compiled native libraries, local reel footage, and package archives are intentionally excluded by `.gitignore`. They are reproducible development outputs rather than release requirements. The retained repository contains the exact production weights, random initial checkpoint, verified result records, replay traces, source, tests, and reel plan.

## Reproduce and verify

Python 3, NumPy, Pillow, PyTorch, and a C++17 compiler are required. PyTorch supports the dense experiments; the default value agent uses NumPy and the native shared library.

```bash
python3 -m bot2048 --seed 42 --quiet

python3 -m scripts.animate_value_agent --seed 42 \
  --output animations/final_rl_bot

python3 -m unittest discover -s tests -v
python3 -m scripts.verify_artifacts
python3 -m scripts.package
```

## Repository map

```text
bot2048/env.py             independent standard 2048 environment
bot2048/value_agent.py     certified search-free RL value ensemble
bot2048/fusion.py          79/100 current-board dense neural committee
native/rl_engine.h         fast board mechanics and realistic spawning
native/td_value.cpp        afterstate TD(lambda) learner and inference API
scripts/evaluate_teacher_ensemble.py
                            paired standard-game evaluation
scripts/animate_value_agent.py
                            replay-verified GIF generation
models/rl/final_value_policy.json
                            frozen production configuration and hashes
```

## Interpretation of “100%”

The project meets the concrete 100/100 certification requested here. Standard 2048 remains stochastic. No finite test can prove victory against every possible future tile sequence, and adversarial spawns can create pathological games. The broad evaluation and confidence interval make that limitation explicit while preserving the exact perfect certification result.
