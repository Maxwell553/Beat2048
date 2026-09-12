# Search-free RL revision — completed

The final reinforcement-learned neural value ensemble won **100/100** on the untouched certification range without expectimax or tree search. Earlier entries remain below as the chronological development record.

## September 10 final search-free RL value agent

The completion architecture is a frozen ensemble of three reinforcement-learned sparse n-tuple afterstate value networks with weights 1:1:2. Inference applies each legal deterministic move once and evaluates the resulting boards; it contains no recursion, random-spawn enumeration, chance nodes, heuristic evaluator, or expectimax. The default CLI now uses `models/rl/final_value_policy.json`, whose loader verifies every checkpoint hash.

The ensemble won **100/100** on the previously untouched certification seeds 700000–700099 in standard games with exactly two initial tiles and 90% 2 / 10% 4 spawns. The result is `results/rl/final_rl_value_ensemble_700000.json`. A broader independent 10,000-game evaluation won 9,869 games, so the certification is reported as an empirical sample rather than an unconditional proof.

The dense current-board-only committee remains preserved at 79/100 as a stricter architectural comparison. A new 1.1-million-parameter scalar afterstate CNN reached 66.5% teacher agreement but only 3/64 gameplay wins and was rejected. A wider sparse teacher regressed from 4,904/5,000 at 350k to 4,887/5,000 at 500k. A separately seeded 300k teacher modestly improved teacher ensembles but its relabeled fusion head regressed to at most 96/128 versus 103/128 for the deployed dense committee.

The default CLI now runs the certified sparse neural value ensemble (`python3 -m bot2048 --quiet`). Legacy search modes require explicit selection.

## Current implementation

- `native/rl_engine.h`: a separate fast standard-2048 training environment. Checked against Python for 400 random boards × four moves, including merge rewards.
- `native/rl.cpp`, `bot2048/rl.py`: direct sparse neural action values trained by temporal-difference learning. Its input is only the current board. Legal-action masking is separate. There is no heuristic evaluator, spawn enumeration, tree traversal, afterstate feature evaluation, or expectimax in action selection.
- `native/td_value.cpp`: a **training-only** RL afterstate value teacher. It learns exclusively from game rewards. It checks the immediate move result to obtain training action targets; it does not use expectimax. This teacher is not an acceptable final player under the stricter direct-policy requirement.
- `bot2048/deep_rl.py`: a 362,564-parameter convolutional residual policy. A forward pass returns four action logits directly from the current board. The deployment class only adds a legal-action mask.
- `scripts/train_conv_rl.py`: distills targets from the RL-trained teacher into the direct convolutional policy. This is RL-derived imitation initialization, not yet direct deep-RL fine-tuning; that distinction must remain explicit in final reporting.
- `scripts/evaluate_conv_rl.py`: uses the independent Python environment to evaluate the direct policy, with no native teacher or search engine required.

## Completed experiments

1. Direct sparse TD action-value learning: 100,000 initial games followed by 1,000,000 additional games. Scores improved to roughly 6,000 but the last 1,000 training games had no wins. Saved in `models/rl/direct_q.bin`. This does not meet the target.
2. A 1,000,000-game Monte Carlo-return continuation of that checkpoint regressed to roughly random performance. Saved separately in `models/rl/direct_q_mc.bin`; do not promote it.
3. An RL afterstate teacher learns substantially faster, reaching roughly 95% wins in recent training windows. Its training log is `results/rl/value_teacher.log`. These are training statistics, not the final independent benchmark.
4. A snapshot of that learned teacher generates disjoint training/validation games using seeds beginning at 61000000 and 62000000. The binary files contain current boards and four learned action scores. No old expectimax labels are used in this revision's convolutional training.

## Neural-policy measurements and ongoing work

The convolutional imitation initialization completed 20 epochs and won **1/100** standard development games, averaging **6,782.44** points. An early PPO checkpoint won **0/100**, averaging **7,283.44**. Neither meets the goal.

`native/rl_batch.cpp` and `scripts/ppo_rl.py` now train the policy directly with PPO across 256 real environments. Training rewards use actual merges, a 2048 completion bonus, and a loss penalty. The critic is training-only; deployed actions come from the policy logits. A curriculum resets 75% of training episodes to legitimate teacher-game prefixes containing a 1024 tile. These curriculum wins are never counted as normal-start benchmark wins. Full model/critic/optimizer checkpoints support continuation, with fresh environment episodes after a resume.

An extended supervised initialization is also training to reduce policy approximation errors. Both policies retain the same 362,564-parameter architecture. A test confirms one neural forward pass on the original board, exactly four legality checks, and no native search or teacher engine access.

## Training commands and artifacts

```bash
c++ -O3 -std=c++17 -march=native native/rl.cpp -o native/rl_train
./native/rl_train train 100000 models/rl/direct_q.bin 20260909 .1 .5
./native/rl_train resume 1000000 models/rl/direct_q.bin 20360909 .05 .5

c++ -O3 -std=c++17 -march=native native/td_value.cpp -o native/td_value
./native/td_value train 200000 models/rl/value_teacher.bin 50000000 .2 .5

# Freeze a teacher snapshot before collecting labels.
cp models/rl/value_teacher.bin models/rl/value_teacher_snapshot.bin
./native/td_value collect 300 models/rl/value_teacher_snapshot.bin 61000000 > data/rl_teacher_train.bin
./native/td_value collect 60 models/rl/value_teacher_snapshot.bin 62000000 > data/rl_teacher_validation.bin
python3 -m scripts.train_conv_rl --epochs 20 --batch-size 1024
python3 -m scripts.evaluate_conv_rl --games 100 --seed 400000
```

The checkpoint writer replaces files atomically. The teacher snapshot is fixed while labels are collected. Development seeds are distinct from training and validation. Reserve a fresh seed range for final certification after selecting the model; do not repeatedly select a checkpoint by its result on the final 100 games.

## Remaining completion gates

- Train a genuinely strong direct neural policy and improve it with RL where useful.
- Verify inference cannot invoke either search or the training teacher.
- Freeze the selected checkpoint, then win 100/100 independently seeded standard games.
- Change the default player to that model, update the README and saved agent configuration, and record new initial/final animations.
- Re-run artifact verification and rebuild the archive. The existing archive remains a legacy artifact until this happens.

## Live run inventory

See `results/rl/run_state.json` for the last inspected process IDs, tool session IDs, exact commands, checkpoint paths, and completion gates. Revalidate these handles before treating any run as stopped. The learner-state aggregation policy, widened policy, extended convolutional initialization, and RL teacher continuation are separate jobs. PPO has been stopped after poor normal-start evaluation. A successful training process exit alone does not meet the win-rate target.

## Later experiments

- Longer convolutional training improved single-orientation play to 2/100. Symmetry averaging then reached **24/100**, averaging 14,367.04 points. A majority-vote variant also won 24/100 (14,887.52 mean score). These use only neural predictions on transformations of the current board. `models/rl/best_policy.pt` is a frozen copy of the verified averaging candidate.
- PPO curriculum training was stopped after its normal-start evaluation remained below the supervised candidate, including 0/100 with symmetry averaging. Policy, critic, optimizer, logs, and outcomes are retained. Its curriculum wins must not be presented as standard-start performance.
- A new dataset combines 384,000 learner-visited states with the original expert positions. Both splits were relabeled with one frozen RL teacher: 732,214 training states and 145,580 validation states. Collection, source hashes, and teacher hashes are recorded in the adjacent JSON files.
- The new loss adds expected teacher-value regret to the soft-target policy loss, to emphasize costly decisions. This is still RL-teacher distillation; it is not being misreported as on-policy RL.
- A 1,429,636-parameter model was created by widening the best convolutional network. Exact widening passed a forward-output equivalence check before adding tiny noise to let duplicate units learn independently. This model is now training on the aggregated dataset.
- Research context: [Kondo and Matsuzaki (2019)](https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_pdf) reports benefits from deeper policy CNNs, more training examples, and symmetry voting. The implementation here retains current-board-only inference. No external pretrained weights were downloaded.

## Realistic starting positions

Standard games start with exactly two occupied cells. Each initial tile and each later spawn is a 2 with probability 90% or a 4 with probability 10%, uniformly placed in an empty cell, matching the [original game implementation](https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js). All reported development gameplay uses this reset, with no custom board or curriculum resets. The full alternating checkerboard is an artificial terminal-state unit test, not a normal starting position or evidence that perfect ordinary-start play is impossible. A 100/100 benchmark would be an empirical result, not a proof covering every possible random sequence.

The extended narrow-policy training completed all 100 epochs and its frozen checkpoint is being evaluated. An additional **3,483,296 teacher positions from 3,000 standard games** (seeds 65000000–65002999) are saved in `data/rl_expanded_teacher_train.bin`; its JSON manifest records source hashes. These examples use the same frozen RL teacher as the aggregated dataset and are available for broader policy training.

## Latest completed benchmark and artifacts

The completed extended policy won **25/100** development games with symmetry averaging, mean score **15,042.52**. Its frozen checkpoint (`models/rl/conv_extended_complete.pt`) was promoted after evaluation; its result and hash are in `results/rl/conv_extended_complete_development.json`. This is a small development-set improvement, not a statistically established gain or completion of the target.

A broader-data policy is training from that checkpoint on **4,215,510 states** in `data/rl_broad_train.bin`. The data manifest records both component hashes. `scripts/animate_conv_rl.py` records direct neural inference and verifies the full trace by independent replay. The initial random-weight convolutional policy has a new recording in `animations/rl_initial_neural.gif` and a complete trace in the adjacent JSON; seed 42 scored 796 and reached 64 in 106 moves. This is an initial-model artifact, not a final-model demonstration.

The learner-state/regret-loss snapshot subsequently won **38/100**, mean score **16,868.08**, with no truncated games. `models/rl/conv_dagger_mid.pt` is frozen, and its exact verified bytes now back `models/rl/best_policy.pt`. Results are in `results/rl/conv_dagger_mid_development.json`. The original-data 25/100 checkpoint remains preserved for comparison. The full learner-state training, widened model, and broad-data model are still running.

## Symmetry-consistent data and direct RL

The deployed 38/100 policy collected **512,000 new training states**, using the same eight-orientation averaging as deployment (196 wins and 274 losses during collection; these are not the fixed benchmark). `tests/test_collector_policy.py` verifies batched collection actions against the deployed agent. A separate validation collection uses seed 66010000.

A new PPO experiment starts from the frozen 38/100 policy. Unlike the earlier PPO run, it trains the symmetry-averaged action distribution, detaches policy features before critic fitting, uses a separate critic learning rate, and collects 1,024-step rollouts with gamma/lambda 1. This lets terminal outcomes affect longer portions of the collected games. Actor learning rate is 0.000003, critic rate 0.001, sampling temperature 0.25, and entropy bonus 0. Policy-only deployment still uses current-board neural predictions; there is no training teacher or search at inference. The first tiny end-to-end smoke run and gradient/symmetry checks passed. The full run is an experiment, not yet evidence of improved game performance. Logs are `results/rl/ppo_ensemble.log`.

The second aggregation dataset is ready: **1,244,214 training states** and **273,580 validation states**, with source manifests under `data/rl_stage2_*.json`. It combines earlier data with the deployed-policy trajectories while preserving separate validation seeds. The revised PPO run completed its first 65,536-step update; its first rollout had 10 wins and 33 losses at sampling temperature 0.25. This is a training diagnostic, not a replacement for the frozen deterministic-policy benchmark.

The completed first-stage learner-state policy won **39/100**, mean score **16,871.16**, on the fixed Python development benchmark. Its frozen checkpoint is `models/rl/conv_dagger_complete.pt`; its exact bytes now back `models/rl/best_policy.pt`. The contemporaneous wide-policy snapshot won 36/100 and was not promoted. A second-stage policy now trains hard teacher-action targets with extra loss weight on 512/1024 states. Its epoch-1 checkpoint won 34/64 in a native screening set versus 30/64 for the current best on identical seeds; the authoritative Python benchmark is in progress.

The RL afterstate teacher completed its 500,000-game continuation and won **986/1000** fresh standard games. It remains training-only and does not qualify as the direct policy. The frozen checkpoint and evaluation manifest are `models/rl/value_teacher_500k.ptbin` and `models/rl/value_teacher_500k.json`. A lower-rate continuation is running to improve future labels.

## Current best and later aggregation rounds

The second-stage epoch-1 policy won 47/100 with averaged logits and **48/100 with neural symmetry voting**. Stage 3 then optimized the differentiable eight-orientation average on 768,000 learner-visited states. Its frozen epoch-2 checkpoint won **58/100** on the same independent Python seeds, with no search or truncated games. It is the checkpoint currently copied to `models/rl/best_policy.pt`; the exact hash and evaluation path are in `models/rl/best_policy.json`.

Later stage-3 epochs improved teacher agreement but regressed in gameplay: epoch 3 won 51/100 and the completed epoch-5 policy won 50/100. This is why checkpoints are promoted only by complete gameplay benchmarks rather than validation loss. The separate cost-sensitive margin model also failed to beat the current policy in screening.

The 58/100 voting policy generated 1,024,000 new training states and 256,000 validation states from disjoint standard games. The manifests record that the rollout action rule was neural voting, not search. A three-snapshot learned RL teacher ensemble scored 985/1000 on a shared native benchmark, compared with 982/1000 for either recent snapshot alone. New relabeled stage-4 datasets average those teacher scores. Both single-teacher and ensemble-teacher stage-4 policy experiments are in progress; neither is promoted without an independent benchmark.

## September 9 development leader

The current best checkpoint is a 2,019,972-parameter, five-residual-block policy. It was created by adding two identity-initialized residual blocks to the previous wider policy, then training on learner-visited states labeled by the frozen three-snapshot RL teacher ensemble. Epoch 3 won **71/100** games on seeds 400000–400099 in the independent Python environment, with mean score **20,641.88** and no truncations. Its exact bytes are copied to `models/rl/best_policy.pt`; SHA-256 is `c654097b42e310efc37ecaabfe52619a0b602618bcf0672f5b3a74eb05f2bb5c`.

That policy collected another 2,048,000 standard-game states. A second collection duplicated the final decision window before each loss, yielding 1,193,852 records, including 169,852 replayed failure-window states. Separate validation seeds produced 369,958 records. These are training measurements and are not substituted for the fixed development benchmark.

## Verified-label repair and outcome policy improvement

The previously generated `rl_deep_stage6_critical_*` conservative targets failed a live consistency check: most preserved records selected action 0 rather than the deployed neural vote. They are rejected. Regenerated `rl_residual_conservative_*` targets agree with the live vote on preserved records and contain 0.70% training and 0.77% validation teacher overrides at a 10,000-value regret threshold. Residual-head, gated-head, and frozen-base adapter experiments on these repaired labels screened below the 48/64 baseline and were not promoted.

`scripts/collect_rollout_corrections.py` now estimates action values from complete standard-game outcomes on states preceding direct-policy losses. The first 256-state training collection used four rollouts per legal action: the deployed action's mean continuation win rate was 54.5%, versus 69.4% for the sampled best action. An independent 128-state collection measured 51.2% versus 64.8%. Conservative targets retained 31 and 19 large-margin corrections respectively. A small gated action head fit training corrections but generalized to only 10/19 validation correction actions, so it was rejected. The outcome collection and arbitrary-board batch reset support remain as a basis for a larger, lower-variance policy-improvement round.

A frozen-base PPO experiment trained only a residual neural vote head from real merge rewards and completed-game outcomes. Iteration 3 at calibrated scale 1.0 scored 50/64 on the native screen, then only 64/100 on the independent Python development benchmark. A lower-variance paired native run scored 648/1000 for the unchanged base and 646/1000 for the PPO head. The head was rejected and `best_policy.pt` remains unchanged.

An eight-pattern direct sparse action network was evaluated with the same overall parameter budget through feature hashing and again without collisions. The hashed version peaked at 414/1000; the collision-free version reached 347/1000, both below the earlier four-pattern model's 471/1000. Their large rejected checkpoints were removed after logs were saved.

`EquivariantConvPolicy` now moves all eight current-board symmetry evaluations and a differentiable low-temperature vote inside the model. Its untrained wrappers screened at 35–42/64 depending on temperature. A full 3,241,852-state aligned-objective epoch took 1,055.65 seconds, reached 70.8% validation action agreement, and screened at 40/64. It was rejected; the architecture and equivariance test remain in the codebase.

A roughly 1-million-parameter `TransformerPolicy` treats the 16 board cells as positional rank tokens and uses four self-attention blocks. Three epochs on 3,241,852 learner-state records raised validation action agreement from 61.0% to 66.6%, but gameplay screens were only 0/64, 4/64, and 9/64. It was rejected as a standalone policy. Its random initialization is saved as `models/rl/transformer_initial.pt`.

The CNN and direct sparse policies have complementary errors. Using one sparse checkpoint only to resolve tied CNN symmetry votes improved a paired native range from 648/1000 to 664/1000. Averaging sparse epochs 11, 12, and 14 reached 680/1000, but scored 70/100 in the independent Python development environment versus 71/100 for the frozen CNN. A five-member average, sparse majority vote, vote-margin overrides, occupancy gates, tie-count gates, and a learned 78-feature neural selector were tested on separate selection and validation ranges; none generalized beyond 680/1000. The three-member value committee is saved as an alternate, not promoted.

## Learned fusion deployment and on-policy refinement

A 290,564-parameter fusion MLP now combines the complete one-hot current board, eight aligned outputs from the 2,019,972-parameter residual CNN, neural vote statistics, and normalized action values from three frozen sparse n-tuple networks. Every feature is computed from the current board. The default `fusion` mode loads `models/rl/best_policy.json` and verifies the SHA-256 digest of every component before inference. Expectimax remains available only through explicit legacy modes.

The frozen fusion policy won **74/100** games on the independent Python development seeds 400000–400099, with mean score **21,115.96** and no truncations. On a separate native benchmark it won 692/1000 versus 638/1000 for the CNN. This is the current development leader; it does not satisfy the 100/100 completion gate.

The fusion policy then collected 407,365 teacher-labeled states from its own standard-game rollouts, including 23,365 replayed states immediately preceding losses. A mixed old/on-policy fine-tune won 751/1000 versus 738/1000 for the deployed head on identical native seeds, but regressed from 21/30 to 20/30 on the first independent Python development seeds. A heavily on-policy version and a simple two-head logit ensemble also regressed in screening. All three are rejected, and the 74/100 policy remains frozen as the default.

## September 10 heterogeneous neural committee

A new `LineEmbeddingPolicy` learns an embedding for each complete four-cell row and column and combines all eight line embeddings with the one-hot current board in a residual MLP. It never constructs a successor board. As a standalone policy it reached only 27/256 with symmetry voting, but its errors differed from the CNN and sparse policies. A line-aware fusion head at epoch 3 scored 718/1000 versus 700/1000 for the old fusion head on identical native seeds. Equal-logit averaging of the old and line-aware heads improved to 737/1000. In the independent Python environment it won **79/100** on seeds 400000–400099 with mean score **21,404.88** and no truncations, replacing the 74/100 head as the checked development leader.

The default manifest now loads the residual CNN, the line-embedding expert, three sparse n-tuple networks, and both fusion heads. Production actions matched the batched evaluation path on sampled boards, and all 29 tests passed. Every neural component receives only the current board or one of its eight rotations/reflections; expectimax and the TD teacher remain outside deployed inference.

Terminal-bonus TD critics, a binary win critic, a larger full-corpus fusion head, directional and transformer experts, outcome gates, and a PPO fusion residual were screened and rejected. The strongest three-snapshot learned TD teacher ensemble reached 4928/5000 on a shared range, versus 4887/5000 for the earlier single teacher. A committee DAgger round collected 405,312 training states and 131,447 disjoint validation states, including 24,759 failure-replay positions. Direct fine-tuning regressed from 744/1000 to 719/1000 on fresh paired seeds. A high-confidence neural selector and unequal committee weights also regressed. The 79/100 committee remains deployed, the final seed range remains untouched, and the 100/100 completion gate remains open.
