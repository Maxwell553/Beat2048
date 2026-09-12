# 2048 Reel — final script and edit plan

Target: **30–34 seconds**, vertical 1080×1920 at 30 fps. The pacing and structure follow the final Snake v14 reel: result first, before/after contrast, one short technical explanation, proof, GitHub, and the series CTA.

## Final spoken script

Record each numbered block as a separate clip.

1. **“I built an AI model that wins at 2048.”**

2. **“This is the same kind of neural agent before training—and this is the final agent afterwards.”**

3. **“I trained it with reinforcement learning, where it played hundreds of thousands of games and learned which boards lead to bigger tiles.”**

4. **“There’s no expectimax and no tree search. The learned neural value model chooses up, down, left, or right.”**

5. **“The final bot won one hundred out of one hundred untouched test games.”**

6. **“The code and trained model are free on my GitHub. This is 2048 in my thirty-games-in-thirty-days challenge. Follow to see the next one.”**

Spoken length: about 86 words. Read naturally at roughly 160–170 words per minute. Do not race the technical sentence.

## Edit timeline

| Time | Spoken block | Visual |
| --- | --- | --- |
| 0.0–3.2 s | 1 | Open immediately on the final replay at 1024 + 1024, then show the real merge into 2048. Put your face in a clean upper-picture cutout or cut to it after the merge. On-screen text: **“I trained AI to beat 2048”**. |
| 3.2–7.5 s | 2 | Side-by-side or hard-cut comparison: `rl_initial_neural.gif` labeled **Before training**, then `final_rl_bot.gif` labeled **After training**. Use actual traces. |
| 7.5–14.0 s | 3 | Start with your face, then zoom out from one animated 2048 board into a wall of many replay tiles, echoing Snake v14’s many-board RL shot. Label it **“Hundreds of thousands of training games”**. The wall is an illustration of scale, not hundreds of archived unique recordings. |
| 14.0–20.0 s | 4 | Show four large direction labels around a readable board: ↑ ↓ ← →. Briefly show a clean code crop from `native/td_value.cpp`. On-screen text: **“RL value network • No expectimax”**. |
| 20.0–24.5 s | 5 | Use the real certification result card: **100 / 100 wins** and smaller text **“Untouched standard-game seeds”**. Keep the qualification readable. |
| 24.5–32.0 s | 6 | Return to your face. Add a small GitHub icon and repository name `Maxwell553/Beat2048`. End on **“Game 2 / 30”** and **“What should I beat next?”** |

Use hard cuts, three restrained punch-ins, clean sentence-case captions, and the existing 2048 tile palette. Keep captions within approximately x=90–990 and y=250–1500. Use two lines maximum and avoid word-by-word bouncing.

## Exact footage to record

Record **two takes of each block**, plus room tone. This should take about 20 minutes.

- Phone vertical and stationary at eye level.
- Use the normal 1× camera, 4K/30 fps if available; otherwise 1080p/30 fps.
- Frame from mid-chest upward with your eyes roughly one-third from the top.
- Face a window or soft lamp. Avoid a bright window behind you.
- Look into the lens, not at your own preview.
- Leave two silent seconds before and after every line.
- Take 1 should be relaxed and conversational. Take 2 should be slightly faster and more energetic.
- If you stumble, pause and repeat the complete block.
- Record ten seconds of room tone without moving the phone or microphone.
- Keep the original files untrimmed and unfiltered. Do not add captions, music, stabilization, or portrait blur.

Name the clips:

```text
A01_hook_take1.mov
A01_hook_take2.mov
A02_before_after_take1.mov
A02_before_after_take2.mov
A03_training_take1.mov
A03_training_take2.mov
A04_no_search_take1.mov
A04_no_search_take2.mov
A05_result_take1.mov
A05_result_take2.mov
A06_cta_take1.mov
A06_cta_take2.mov
ROOM_10s.mov
```

Put them in `reel_01/incoming/`. You do **not** need to record gameplay, the terminal, GitHub, or scrolling code. The repository already contains verified gameplay traces and the result data needed for those shots.

Optional footage, only if convenient:

- One steady six-second over-the-shoulder shot of you watching the final 2048 run.
- One silent reaction shot: look at the screen, then a small satisfied smile when 2048 appears. Avoid a staged celebration.

## Delivery notes

- Use `animations/rl_initial_neural.json` and `animations/final_rl_bot.json` as the authoritative gameplay sources.
- The final trace reaches 2048 in 1,176 moves on seed 42 and is replay-verified.
- The certification card must use `results/rl/final_rl_value_ensemble_700000.json`.
- Say **“100 out of 100 untouched test games,”** not “guaranteed unbeatable.” The broader measured rate is 9,869/10,000.
- “No expectimax” is accurate. The bot applies each legal move once and scores its deterministic result with learned RL value networks; it does not expand random future spawns or a search tree.
- Label sped-up gameplay **“accelerated replay.”** Do not imply that animation speed is inference speed.

## Suggested post caption

I trained an RL agent to beat 2048—without expectimax or tree search.

The final frozen model won 100/100 untouched standard games. Across a broader 10,000-game evaluation it won 98.69%, so the perfect test set is a measured result, not a mathematical guarantee.

The game starts normally with two tiles, and every move is chosen by reinforcement-learned neural value networks.

Code + trained model: github.com/Maxwell553/Beat2048

Game 2 of 30. What should I beat next?

#2048 #GameAI #ReinforcementLearning #MachineLearning
