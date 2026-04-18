# Fine-tuning EmbeddingGemma for Electrical Engineering Information Retrieval

A production-ready, domain-specialized embedding model for electrical and electronics engineering. Fine-tuned from [unsloth/embeddinggemma-300m](https://huggingface.co/unsloth/embeddinggemma-300m) — Unsloth's optimized mirror of Google's [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) — using a LoRA adapter trained with [Unsloth](https://github.com/unslothai/unsloth)'s `FastSentenceTransformer` on the [ElectricalElectronicsIR](https://huggingface.co/datasets/disham993/ElectricalElectronicsIR) dataset, then exported to GGUF for efficient deployment via `llama.cpp`.

**If you are building semantic search, RAG, or a knowledge base over electrical engineering content, this model gives you near-perfect retrieval at the size and speed of a 300M parameter encoder.**

---

## Results

Evaluated on the held-out test split (2,000 queries) of ElectricalElectronicsIR using `sentence_transformers.evaluation.InformationRetrievalEvaluator`. See `Evaluate_All_Models.ipynb` for the full evaluation code.

| Model | MAP@100 | NDCG@10 | MRR@10 | Recall@10 |
|---|---|---|---|---|
| `google/embeddinggemma-300m` (baseline) | 0.5753 | 0.6221 | 0.5682 | 0.7925 |
| `unsloth/embeddinggemma-300m` (baseline) | 0.5753 | 0.6221 | 0.5682 | 0.7925 |
| **LoRA adapter** ⭐ | **0.9795** | **0.9847** | **0.9795** | **1.0000** |
| **Merged 16-bit** ⭐ | **0.9797** | **0.9849** | **0.9797** | **1.0000** |
| **GGUF f16** ⭐ | **0.9849** | **0.9887** | **0.9849** | **0.9995** |
| **GGUF q8_0** ⭐ | **0.9844** | **0.9883** | **0.9844** | **0.9995** |
| **GGUF q4_k_m** ⭐ | **0.9841** | **0.9879** | **0.9840** | **0.9990** |
| **GGUF q5_k_m** ⭐ | **0.9824** | **0.9866** | **0.9823** | **0.9990** |

![Retrieval performance comparison across all model variants](https://huggingface.co/disham993/electrical-embeddinggemma-ir_finetune_16bit/resolve/main/eeir-models-retrieval-comparison.png)

### Headline findings

- **+41 points of MAP@100, +21 points of Recall@10** over the general-purpose baseline. The fine-tuned model retrieves the correct passage at rank 1 on virtually every query.
- **GGUF q4_k_m nearly matches f16 quality** — quantization down to 4-bit is effectively lossless on this task (MAP delta: 0.0008). Ship q4_k_m for ~4× smaller memory footprint with no meaningful retrieval quality trade-off.
- **LoRA adapter ≈ merged model** — negligible metric gap (MAP delta: 0.0002) confirms a near-lossless merge; either artifact is safe to deploy.

---

## Use Cases

This model is built for any application that needs to match electrical engineering queries to technical passages. Concrete examples:

- **RAG over engineering documentation** — IEEE papers, IEC/IEEE standards, datasheets, application notes, technical manuals, textbooks
- **Enterprise knowledge bases** — internal wikis, design reviews, post-mortems, and field service reports at electrical utilities, OEMs, semiconductor firms, and EPC contractors
- **Semantic search in CAD and PLM tools** — retrieve relevant design notes, test reports, or component specifications from historical project archives
- **Engineering assistants and copilots** — grounding an LLM's answers in a verified corpus of electrical engineering knowledge instead of relying on parametric recall
- **Educational platforms** — matching student questions to the right section of a textbook, lecture transcript, or problem set in EE/ECE coursework
- **Standards compliance tooling** — retrieving the relevant clauses of IEC 61850, IEEE 1547, NEC, etc. given a natural-language description of a design constraint
- **Technical support triage** — routing customer tickets about inverters, drives, protection relays, or test equipment to the correct KB article
- **Patent prior-art search** — finding conceptually similar electrical/electronic inventions even when the query and document use different terminology
- **Literature review assistants** — clustering and retrieving related work across power systems, RF/microwave, VLSI, photonics, control systems, and signal processing
- **On-device / offline applications** — the GGUF q4_k_m build runs on a laptop CPU, making it suitable for field technicians, air-gapped environments, and embedded deployments

### Where it is **not** the right fit

This is a specialist. It will likely underperform the general-purpose baseline on:

- General-web, biomedical, legal, financial, or consumer-product queries
- Code search, multilingual retrieval outside English, or conversational chitchat
- Any domain whose vocabulary and conceptual structure differs substantially from electrical/electronics engineering

For mixed-domain applications, run this model alongside a general-purpose embedder and route queries by topic.

---

## Released Artifacts

All model variants are published on the Hugging Face Hub:

| Artifact | Format | Use Case |
|---|---|---|
| [electrical-embeddinggemma-ir_lora](https://huggingface.co/disham993/electrical-embeddinggemma-ir_lora) | LoRA adapter | Stack on base EmbeddingGemma-300M |
| [electrical-embeddinggemma-ir_finetune_16bit](https://huggingface.co/disham993/electrical-embeddinggemma-ir_finetune_16bit) | Merged fp16 | Sentence Transformers, vLLM, TEI |
| [electrical-embeddinggemma-ir_f16](https://huggingface.co/disham993/electrical-embeddinggemma-ir_f16) | GGUF f16 | Full-precision `llama.cpp` inference |
| [electrical-embeddinggemma-ir_q8_0](https://huggingface.co/disham993/electrical-embeddinggemma-ir_q8_0) | GGUF q8_0 | 8-bit quantization |
| [electrical-embeddinggemma-ir_q4_k_m](https://huggingface.co/disham993/electrical-embeddinggemma-ir_q4_k_m) | **GGUF q4_k_m** | **Recommended for production deployment** |
| [electrical-embeddinggemma-ir_q5_k_m](https://huggingface.co/disham993/electrical-embeddinggemma-ir_q5_k_m) | GGUF q5_k_m | 5-bit quantization |

For loading, inference, and integration examples, see the individual model cards on the Hugging Face Hub linked above.

---

## Training Details

| | |
|---|---|
| **Base model** | `unsloth/embeddinggemma-300m` (308M params) |
| **Method** | LoRA via Unsloth's `FastSentenceTransformer` |
| **LoRA rank / alpha** | r=32, α=64 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Loss** | `MultipleNegativesRankingLoss` (in-batch negatives) |
| **Batch size** | 128 per device × 2 gradient accumulation = 256 effective |
| **Learning rate** | 2e-5 (linear schedule, 3% warmup) |
| **Max steps** | 100 |
| **Max sequence length** | 1024 |
| **Precision** | bf16 (EmbeddingGemma does not support fp16) |
| **Batch sampler** | `NO_DUPLICATES` to avoid false negatives from duplicate anchors |
| **Hardware** | RTX 5090 (also runs on a free Colab T4 with reduced batch size) |

---

## Dataset

[`disham993/ElectricalElectronicsIR`](https://huggingface.co/datasets/disham993/ElectricalElectronicsIR) — 20,000 question-passage pairs covering electrical engineering, electronics, power systems, and communications.

- 16k train / 2k validation / 2k test
- Queries: 133–822 characters; passages: 586–5,590 characters
- Topics include phased array antennas, IEC 61850 protocols, Josephson junctions, OTDR measurements, MIMO channel estimation, FPGA partial reconfiguration, and more

---

## Reproducing the Results

```bash
git clone https://github.com/di37/information-retrieval-electrical-electronics-finetuning.git
cd information-retrieval-electrical-electronics-finetuning
```

### Local / cloud GPU

Install PyTorch first, matching your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Then open `Finetuning_EmbeddingGemma_EEIR_RTX_5090.ipynb` in Jupyter. Note that bf16 training requires an Ampere GPU or newer (RTX 30xx/40xx/50xx, A100, H100, etc.); older GPUs will fall back to fp32 automatically.

To reproduce the evaluation results, open `Evaluate_All_Models.ipynb`. It pulls all model variants directly from the Hugging Face Hub and runs `InformationRetrievalEvaluator` on the held-out test split.

### Google Colab

Open `Finetuning_EmbeddingGemma_EEIR_RTX_5090.ipynb` directly in Colab — the first cell handles Colab-specific installation. A free T4 runtime works with `per_device_train_batch_size = 32` instead of 128.

### Hugging Face authentication

Before pushing artifacts to the Hub, set your token via Colab Secrets or an environment variable. **Do not hardcode it in the notebook.**

```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

---

## Stack

- [**Unsloth**](https://github.com/unslothai/unsloth) — 2× faster LoRA training with 20% less VRAM
- [**Sentence Transformers**](https://sbert.net) — training loop, IR evaluator, encoding API
- [**PEFT**](https://github.com/huggingface/peft) — LoRA implementation
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) / [**llama-cpp-python**](https://github.com/abetlen/llama-cpp-python) — GGUF inference

---

## License

- **Code and notebook** in this repository: [MIT](./LICENSE)
- **EmbeddingGemma model weights**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **ElectricalElectronicsIR dataset**: MIT (see dataset card)

