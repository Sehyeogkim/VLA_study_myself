# Phase 3 — Image encoder + decoder (mini **VLM** track)

Same skeleton as `2_encoder_decoder`: **vision memory → cross-attn decoder → text**.  
Encoder side = patches / ViT (instead of source **text**).

**This folder: text output only — 3 topics below.**  
**VLA (vision → action)** is a separate track (**Vision–Action model**); see `phase2_policy_myself` and a future dedicated phase — not mixed in here.

Task names are **labels** (dataset / prompt / eval), not a rigid official taxonomy.

---

## Scope here (3 only)

| # | Name | I/O | Notes |
|---|------|-----|--------|
| 1 | **Captioning** | `image` → `text` | Start here (e.g. COCO-style). |
| 2 | **VQA** | `image` + `question` → `answer` | Short answers. |
| 3 | **Chat VLM** | `image` + `conversation` → `next reply` | Multi-turn templates. |

Same architecture family; differences are mostly **data format and prompts**.

---

## Not in this folder

| Name | I/O | Where |
|------|-----|--------|
| **VLA** / **Vision–Action** | `image` + `instruction` → `action` | Phase 2 + later **vision–action** phase (flow/diffusion head, sim eval, …). |

---

## Planned order

1. Caption  
2. VQA  
3. Chat VLM  

---

## Stack reminder

```text
Phase 3 #3 (this folder):  image (+ text) → vision encoder → memory → decoder → text
Vision–Action (elsewhere): image (+ text) → memory → cross-attn / expert → actions
```
