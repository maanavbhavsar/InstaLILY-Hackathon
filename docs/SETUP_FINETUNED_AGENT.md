# Setting Up the Fine-Tuned Understanding Agent

You have **two model roles** in FieldTalk:

| Role | What it does | Who trains it | Where it’s used |
|------|----------------|----------------|------------------|
| **Vision / lip reading** | 16 mouth frames → GRID phrase (e.g. "bin blue at f two now") | **Partner** (GRID corpus) | `inference.py` — env **`MODEL`** |
| **Understanding agent** | Phrase + context → priority, actions, reasoning | **You** (200 industrial examples on VM) | `reasoning.py` — env **`AGENT_MODEL`** |

This doc is about the **understanding agent** (200-example Gemma). Lip-reading integration is at the end.

---

## Best option: Get GGUF on the VM, then use Ollama

So the app keeps one stack (Ollama) for both vision and agent.

### 1. On the VM (fix GGUF conversion)

Exit code 127 was because the converter calls `python` and the VM only has `python3`. Fix once:

```bash
sudo ln -sf $(which python3) /usr/bin/python
```

Then re-run training. It will train again (~3 min) and this time GGUF conversion should succeed:

```bash
cd ~   # or wherever finetune_gemma_agent.py and industrial_safety_200.jsonl are
python3 finetune_gemma_agent.py
```

When it finishes you should see:

- `fieldtalk_agent_model_merged/` — merged 16-bit (HuggingFace)
- `fieldtalk_agent_gguf/` — contains a **.gguf** file (name may be like `gemma-3-4b-it-Q4_K_M.gguf` or similar)

### 2. Copy the GGUF to your demo machine

From the VM, copy the GGUF folder (or at least the `.gguf` file) to your project on the machine where you run the app, e.g.:

```bash
# On VM (from the directory that contains fieldtalk_agent_gguf)
scp -r fieldtalk_agent_gguf youruser@your-laptop:"/path/to/InstaLILY Hackathon/"
```

Or use any other copy method (USB, cloud, etc.).

### 3. On the demo machine: create the Ollama agent model

In the project root (where you have `Modelfile.agent` and the copied `fieldtalk_agent_gguf/`):

```powershell
# List the .gguf file (name may vary)
dir fieldtalk_agent_gguf

# If the file is e.g. gemma-3-4b-it-Q4_K_M.gguf, edit Modelfile.agent and set:
#   FROM ./fieldtalk_agent_gguf/gemma-3-4b-it-Q4_K_M.gguf
# (or whatever the actual filename is). Then:

ollama create fieldtalk-agent -f Modelfile.agent
```

### 4. Run the app with the fine-tuned agent

```powershell
$env:AGENT_MODEL = "fieldtalk-agent"
python -m streamlit run app.py
```

The **understanding** step (phrase + context → actions) now uses your 200-example Gemma. The **lip reading** step still uses whatever vision model is in `MODEL` (e.g. base `gemma3:4b`) until you switch it to the partner’s model.

---

## If you can’t get GGUF on the VM

- **Option A:** Convert on another machine that has `python` and the same Unsloth/llama.cpp setup. Copy `fieldtalk_agent_model_merged/` from the VM, then run the same GGUF conversion (e.g. a small script that loads that merged model and calls `save_pretrained_gguf`).
- **Option B:** For the demo only, keep using the **MLP** (`FINE_TUNED_AGENT_PATH=models/finetuned_agent.pt`). You can still say: “We also fine-tuned Gemma 3 on 200 industrial examples on a VM for the agent; that model is for production / we’re loading it via Ollama when the GGUF is ready.”

---

## Integrating the partner’s GRID lip-reading model

When your partner delivers the **Gemma fine-tuned on the GRID corpus** for lip reading:

1. **If they give you an Ollama model name** (e.g. they ran `ollama create grid-lip-reader -f ...`):
   - Set the **vision** model in the app to that name:
   ```powershell
   $env:MODEL = "grid-lip-reader"
   ```
   No code change needed: `inference.py` already uses `MODEL` for the vision call.

2. **If they give you a GGUF or HuggingFace checkpoint**:
   - They (or you) create an Ollama model from that GGUF (Modelfile + `ollama create`), then set `MODEL` to that model name as above.
   - Or you add a second path in `inference.py` that loads their model via HuggingFace/transformers and runs inference on the 16-frame image; that’s a larger code change and only needed if you don’t put the model into Ollama.

So:

- **Lip reading** = `MODEL` (partner’s GRID model when ready).
- **Understanding agent** = `AGENT_MODEL` (your 200-example Gemma; best option is GGUF in Ollama as above).

That keeps the split clear and the setup repeatable.
