# Qwen2.5-0.5B Pretraining: "What Next" & Cheat Sheet

This guide documents the exact commands, workflow, troubleshooting tips, and post-training steps for pretraining **Qwen2.5-0.5B** on **10B tokens of FineWeb-Edu** using an NVIDIA H100 GPU.

---

## 1. Quick Summary of Epoch 2 Run

| Setting | Value | Notes |
| :--- | :--- | :--- |
| **Model** | Qwen2.5-0.5B (~494M params) | Vocab: 151,936, Layers: 24, GQA: 14 Q / 2 KV heads |
| **Starting Checkpoint** | `checkpoints/qwen_0_5b_final.pt` | Epoch 1 weights preserved |
| **Starting Step** | **Step 18,878** | Learning rate scheduler continues decaying smoothly |
| **Dataset Stream** | **Sample 0 (Restarted for Epoch 2)** | Full second pass over the dataset |
| **Target Epochs** | **5** | Epoch limit removed; training continues until max_steps |
| **Checkpoint Interval** | **Every 1,000 steps** | Saved to `./checkpoints/step_<N>.pt` (~every 34 mins) |
| **Checkpoint Retention** | **Keep last 5** | Conserves disk space while protecting progress |
| **Final Saved Model** | `qwen_0_5b_final_epoch2.pt` | Original `qwen_0_5b_final.pt` is never overwritten |

---

## 2. Linux & tmux Cheat Sheet

### Check if You Are Inside `tmux` or Regular `bash`
```bash
echo $TMUX
```
* **Inside tmux:** Prints a socket path like `/tmp/tmux-983/default,12345,0` (and a green bar appears at the bottom).
* **Outside tmux:** Prints a blank line.

---

### Attaching & Detaching

| Action | Command / Keypress |
| :--- | :--- |
| **Detach (leave training in background)** | Press <kbd>Ctrl</kbd> + <kbd>B</kbd>, release, then press <kbd>D</kbd> |
| **Re-attach to training session** | `tmux attach -t pretrain` |
| **Force attach (if already attached elsewhere)**| `tmux attach -d -t pretrain` |
| **List running sessions** | `tmux ls` |
| **Kill / Terminate session completely** | `tmux kill-session -t pretrain` |

---

### Screen Management (Fixing Splits / Clutter)

| Problem / Goal | Solution |
| :--- | :--- |
| **Screen is split into narrow columns / mess** | Run `tmux kill-pane -a` *(instantly kills all other panes except current)* |
| **Zoom in / make active pane full screen** | Press <kbd>Ctrl</kbd> + <kbd>B</kbd>, release, then press <kbd>Z</kbd> *(press again to unzoom)* |
| **Close single pane manually** | Type `exit` and press <kbd>Enter</kbd> |

---

## 3. How to Monitor Training

### Option A: From inside `tmux`
Just look at the terminal output. It logs progress every 20 steps:
```text
Epoch  |   Step   |    Loss    | Perplexity  |  Tokens/s  |     LR
--------------------------------------------------------------------------------
  1    |    20    |  11.8067   |  134145.55  |   64266    |  4.20e-06
  1    |   120    |   9.1542   |   9453.98   |   64291    |  2.42e-05
```

### Option B: From outside `tmux` (without attaching)
```bash
# View the live metrics log (updates automatically)
tail -f ~/openlanguagemodel/examples/qwen2_5-0_5b-fineweb-edu-10b/logs/metrics_*.jsonl

# Quick snapshot of the last 10 steps
tail -n 10 ~/openlanguagemodel/examples/qwen2_5-0_5b-fineweb-edu-10b/logs/metrics_*.jsonl

# Peek directly into the tmux screen to see the live table
tmux capture-pane -pt pretrain -S -25

# Check GPU utilization & temperature (target: 90%+ util, ~41 GB VRAM)
watch -n 1 nvidia-smi
```

### Healthy Signs to Look For
* **Loss:** Drops monotonically from ~11.93 (random init) down towards ~2.8 – 3.2 by the end of training.
* **Perplexity:** Drops steadily from ~140,000 down into double digits.
* **Tokens/s:** Sustained at ~64,000+ tokens/sec without stalling.
* **Learning Rate:** Climbs from 0 to peak $3.0 \times 10^{-4}$ at Step 1,500, then smoothly decays along cosine schedule.

---

## 4. Checkpoints & Resumption

### How Checkpoints Work
* Checkpoints are written to `./checkpoints/step_<N>.pt` every 2,500 steps (~327M tokens).
* `keep_last_n: 3` ensures only the 3 most recent checkpoints are kept on disk (~2.5 GB each).
* **Old checkpoints DO NOT ruin anything.** Checkpoints are never automatically loaded on launch; they are only loaded if you explicitly pass `--resume`.

### How to Resume If Training Is Ever Interrupted
If the server reboots or connection drops unexpectedly:
1. Re-enter directory:
   ```bash
   cd ~/openlanguagemodel/examples/qwen2_5-0_5b-fineweb-edu-10b
   ```
2. Find the latest saved checkpoint:
   ```bash
   ls -lht checkpoints/
   ```
3. Resume training from that checkpoint:
   ```bash
   python -u train.py --config config.yaml --resume checkpoints/step_<LATEST_STEP>.pt
   ```
   *(The script automatically restores model weights, optimizer states, AdamW momentum, LR scheduler position, and fast-forwards the dataset to the exact sample).*

---

## 5. Cooldown Phase (Annealing for Final Epoch 2 Completion)

Since training terminates after **Epoch 2 (~Step 37,756)**, switching to a smooth cooldown decays the learning rate from the active rate (~$2.5 \times 10^{-4}$) down to `min_lr` ($3.0 \times 10^{-5}$), breaking through the ~3.0 loss floor down to ~2.8x.

### Launch Cooldown Command:
```bash
python -u train.py \
    --config config.yaml \
    --resume $(ls -t checkpoints/emergency_step_*.pt checkpoints/step_*.pt | head -n 1) \
    --cooldown \
    --cooldown_target_step 37756 \
    --save_every 1000 \
    --final_name qwen_0_5b_final_epoch2.pt
```

---

## 6. What Next: When Training Completes (~37,756 Steps)

Once training finishes, `qwen_0_5b_final_epoch2.pt` will be saved to `./checkpoints/`.

### Step 1: Verify Final Checkpoint
```bash
ls -lh checkpoints/qwen_0_5b_final_epoch2.pt
```

### Step 2: Test Text Generation (Inference)
Run test prompts to see the model generate text:
```bash
python inference.py \
  --checkpoint checkpoints/qwen_0_5b_final_epoch2.pt \
  --prompt "The future of artificial intelligence is" \
  --max_new_tokens 150 \
  --temperature 0.7
```

### Step 3: Download Checkpoints to Laptop (CRITICAL Before Server Closes!)
Before your 20-hour GPU rental window expires, download your final weights to your laptop:

Run this command **on your laptop terminal** (PowerShell / macOS / Linux):
```powershell
# Epoch 2 Final Model
scp -P 21510 team09@global.prd.ga.run.brev.nvidia.com:openlanguagemodel/examples/qwen2_5-0_5b-fineweb-edu-10b/checkpoints/qwen_0_5b_final_epoch2.pt ./

# Download metrics logs
scp -P 21510 -r team09@global.prd.ga.run.brev.nvidia.com:openlanguagemodel/examples/qwen2_5-0_5b-fineweb-edu-10b/logs ./
```
