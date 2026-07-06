# SO-101 checkpoints worth running (July 2026)

Survey of publicly available SO-101 policy checkpoints on the Hugging Face Hub, aimed at
running inference locally on an RTX 5090 (32 GB) through LeRobot / `inspect-robots-so101`.

## TL;DR

1. **Best zero-shot bet: [`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101)** —
   the only *generalist* SO-100/101 checkpoint, fine-tuned on a community SO-arm mixture with
   absolute joint-pose control and language instructions. Use the LeRobot-format conversion
   [`lerobot/MolmoAct2-SO100_101-LeRobot`](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot)
   (~21 GB weights, fits the 5090) to run in-process via this repo's `lerobot` policy.
2. **Best expected success on *your* arm: fine-tune a base model on ~50 episodes of your own
   teleop data.** [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (450M, pretrained
   on SO-100/101 community data) is the community default; [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)
   is the strongest. Both train comfortably on a 5090.
3. **Community fine-tunes are rig-specific.** They demo well but were trained on someone else's
   arm + camera placement; expect them to need your cameras to match theirs closely, or to
   serve mainly as smoke-test / repro material.

## Reality check: why "really good SO-101 checkpoint" is subtle

Almost every SO-101 checkpoint on the Hub is an imitation-learning fine-tune on **one person's
rig**: their camera mounts, lighting, table, calibration, and task objects. ACT and small VLA
policies memorize that visual context — moving to a different SO-101 usually drops success to
near zero. The practical hierarchy:

- **Generalist SO-arm checkpoints** (trained across many rigs) transfer *partially* — worth
  trying zero-shot, expect degraded-but-nonzero behavior.
- **Base VLAs** transfer *knowledge*, not behavior — they need a short fine-tune on your data
  but then routinely hit 70–90% on pick-and-place.
- **Single-rig fine-tunes** transfer essentially nothing unless you clone the author's setup
  (some, like the LeIsaac/Isaac-Sim ones, are reproducible because the "rig" is a simulator).

## Tier 1 — Generalist SO-100/101 checkpoints (try zero-shot)

### MolmoAct2-SO100_101 (AllenAI)

| | |
|---|---|
| Repos | [`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101) (transformers), [`lerobot/MolmoAct2-SO100_101-LeRobot`](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) (LeRobot format) |
| Popularity | 4.5k downloads / 18 likes (by far the most-adopted SO-arm checkpoint) |
| Architecture | Molmo2-ER VLM + flow-matching continuous action expert ([paper](https://arxiv.org/abs/2605.02881), [blog](https://allenai.org/blog/molmoact2)) |
| Training | Fine-tuned on an SO-100/101 community mixture, **absolute joint-pose control**, language-annotated |
| Size | ~21 GB safetensors → fits 5090 (32 GB) for inference |
| Notes | Pass `norm_tag="so100_so101_molmoact2"`; camera order does not matter; continuous-action mode recommended. The LeRobot repo ships the full `policy_preprocessor`/`policy_postprocessor` pipeline, so it loads with `lerobot`'s factory and works with this repo's in-process `LeRobotPolicy`. |

This is the SO-ARM sibling of the MolmoAct2 checkpoints that `inspect-robots-yam` uses for YAM —
same family, but runnable in-process here. Its absolute-joint contract matches this package's
declared `joint_pos` control mode (no `joints_are_delta` needed).

## Tier 2 — Base models to fine-tune on your own data (highest real success)

| Model | Params / weights | Why |
|---|---|---|
| [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | 450M / 0.9 GB | Pretrained on LeRobot community SO-100/101 data ([SmolVLA blog](https://huggingface.co/blog/smolvla)); the default SO-101 fine-tune target; trains in hours on a 5090. 49k downloads. |
| [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | ~3.6B / 14 GB | Physical Intelligence π₀.₅ — best open-world generalization of the openly available VLAs; LeRobot-native. 22k downloads. |
| [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base) | ~3.3B | π₀ predecessor; still a strong flow-matching baseline. |
| [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) | — | Newer xVLA family; several SO-101 community fine-tunes exist already. |
| [`nvidia/GR00T-N1.5-3B`](https://huggingface.co/nvidia/GR00T-N1.5-3B) (and N1.6/N1.7) | 3B | NVIDIA's generalist; the [NVIDIA sim-to-real SO-101 course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/04-lerobot.html) targets exactly this arm. Uses Isaac-GR00T tooling rather than plain LeRobot. |
| ACT (train from scratch) | ~50M | Not pretrained, but 50 episodes of your data → often 80–90% on a single task; the cheapest strong baseline ([community write-up](https://huggingface.co/blog/sherryxychen/train-act-on-so-101)). |

Recommended recipe for "really good" on your hardware: record ~50 episodes with
`lerobot-record`, fine-tune `smolvla_base` (fast) **and** `pi05_base` (strong) on the 5090,
compare.

## Tier 3 — Notable community SO-101 fine-tunes

Useful as references, smoke tests, or if you replicate their setup. Downloads as of 2026-07-05.

| Checkpoint | Policy | Task / provenance | DLs |
|---|---|---|---|
| [`anikitakis/vla_so101_pick_n_place_full_expert`](https://huggingface.co/anikitakis/vla_so101_pick_n_place_full_expert) | SmolVLA | Pick-and-place, 3-cam **DAgger** dataset (expert corrections) | 1015 |
| [`edge-inference/smolvla-so101-pick-orange`](https://huggingface.co/edge-inference/smolvla-so101-pick-orange) | SmolVLA | 3-orange pick → plate in **LeIsaac (Isaac Sim)** — reproducible since the rig is simulated | 827 |
| [`Sa74ll/smolvla_so101_pickandplace`](https://huggingface.co/Sa74ll/smolvla_so101_pickandplace) | SmolVLA | Trained on the **official** [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) dataset; claims 87.7% *per-joint validation* success (offline metric, not real-robot rollouts) | 101 |
| [`Jingyi-Z/pi0_so101`](https://huggingface.co/Jingyi-Z/pi0_so101) | π₀ | Pick-place fine-tune, most-liked π₀ SO-101 | 159 |
| [`hjkso1406/pi05-so101-4tasks-aug`](https://huggingface.co/hjkso1406/pi05-so101-4tasks-aug) | π₀.₅ | **Multi-task** (4 tasks × 100 eps, augmented) — rare multi-task SO-101 checkpoint | 76 |
| [`xBerry/pi05_so101_sim2real`](https://huggingface.co/xBerry/pi05_so101_sim2real) | π₀.₅ | Sim-to-real transfer experiment | 53 |
| [`binhpham/sim_so101_cubes_act_small`](https://huggingface.co/binhpham/sim_so101_cubes_act_small) | ACT | Cube stacking in sim | 136 |
| [`whosricky/xvla-so101-megamix-v1`](https://huggingface.co/whosricky/xvla-so101-megamix-v1) | xVLA | Trained on the [`so101-megamix-v1`](https://huggingface.co/datasets/whosricky/so101-megamix-v1) multi-task mix | 9 |
| [`Vizuara/dreamzero-so101-lora`](https://huggingface.co/Vizuara/dreamzero-so101-lora) | DreamZero WAM | **World Action Model** (Wan2.1-I2V-14B + action heads): predicts 24 actions *and* 33 video frames jointly. Most-liked SO-101 repo (8 likes). Research-grade; 14B base is tight-to-infeasible on one 5090 without offloading, and it needs the DreamZero codebase, not LeRobot. | 26 |

Not in LeRobot format / separate tooling: the various GR00T fine-tunes
(`ArturFrost/gr00t_so101-ball-box-*`, `sreetz-nv/gr00tn17oss_so101-*`) require Isaac-GR00T.

## Datasets (if you fine-tune)

- [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) — the official SO-101 pick-place dataset from the SmolVLA release.
- [`whosricky/so101-megamix-v1`](https://huggingface.co/datasets/whosricky/so101-megamix-v1) — community multi-task megamix.
- [`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange) — Isaac Sim teleop data (fully reproducible eval env).

## Running on the 5090

- **VRAM:** everything above except DreamZero fits: SmolVLA ~1 GB, π₀/π₀.₅ ~14 GB,
  MolmoAct2-SO100_101 ~21 GB weights + activations (bf16 inference recommended).
- **Blackwell (sm_120):** needs PyTorch ≥ 2.7 with cu128 wheels — current `lerobot` supports
  this; just don't install an old pinned torch.
- **This repo:** all LeRobot-format checkpoints (ACT / SmolVLA / π₀ / π₀.₅ / xVLA / MolmoAct2)
  load through `lerobot`'s policy factory and run in-process behind `LeRobotPolicy` via
  `predict_action_chunk` + the checkpoint's shipped pre/post processors. The postprocessor
  unnormalizes to native motor units — the embodiment commands them verbatim (clamped), per
  this repo's safety invariants.
- **Scoring rollouts:** [`lerobot/Robometer-4B`](https://huggingface.co/lerobot/Robometer-4B)
  ([ROBOMETER docs](https://huggingface.co/docs/lerobot/main/en/robometer)) is a Qwen3-VL-4B
  video-language reward model that predicts per-frame task progress/success — handy as an
  automated judge for eval runs, though this framework's scorer contract still expects
  `termination_reason="success"`.

## Appendix: running DreamZero-SO101 locally

DreamZero is the one entry above with no supported consumer-GPU path — upstream
[dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) targets 2×H100 distributed
inference (CUDA 12.9, torch 2.8, flash-attn). On a single 5090 (32 GB) it is tight but
plausible for **offline** prediction: the Wan2.1-I2V-14B DiT is ~28 GB bf16 and the UMT5-XXL
text encoder ~11 GB, so they can't be resident together — run the text encoder on CPU or
load→encode→free it before the DiT. No fp8/offloading support exists in the DreamZero
codebase itself.

```bash
conda create -n dreamzero python=3.11 -y && conda activate dreamzero
pip install torch --index-url https://download.pytorch.org/whl/cu129
MAX_JOBS=8 pip install flash-attn --no-build-isolation

git clone https://github.com/dreamzero0/dreamzero.git
git clone https://github.com/Vizuara-AI-Lab/dreamzero-so101.git   # model card's "vizuara/" URL is stale
cd dreamzero && pip install -e .
git apply ../dreamzero-so101/patches/so101_embodiment.patch

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
huggingface-cli download Vizuara/dreamzero-so101-lora --local-dir ./checkpoints/dreamzero-so101-lora

python ../dreamzero-so101/scripts/infer_demo.py \
  --model-path ./checkpoints/dreamzero-so101-lora \
  --base-model-path ./checkpoints/Wan2.1-I2V-14B-480P \
  --image ./sample_obs.jpg \
  --prompt "Pick up the red cube and place it in the bowl"
```

Third-party single-box reference: [grmpn/dreamzero-so101-inference](https://github.com/grmpn/dreamzero-so101-inference)
(`offline_inference.py`, takes front/gripper/top images + joint positions). Notes: actions are
**relative** joint positions (use `joints_are_delta=True` in this repo's embodiment); trained
only on 400 episodes / 8 tasks at 320×176; treat as offline prediction, not a real-time
control loop, on one GPU.

## Suggested plan

1. Pull `lerobot/MolmoAct2-SO100_101-LeRobot`, run it zero-shot on a simple pick task
   (language-prompted). This is the only checkpoint with a real chance of working untuned.
2. In parallel, record ~50 teleop episodes of your target task.
3. Fine-tune `smolvla_base` (sanity, fast) and `pi05_base` (quality) on the 5090.
4. Optionally reproduce `edge-inference/smolvla-so101-pick-orange` in LeIsaac if you want a
   checkpoint whose eval environment you can exactly recreate.

---
*Method note: found via Hugging Face Hub API searches (`so101`, `so100`, per-policy-type
queries) sorted by downloads and likes, plus model-card review of the top hits. Download
counts are lifetime; "success rates" on model cards are self-reported and usually offline
validation metrics, not independent real-robot evals.*
