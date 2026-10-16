'''
=====
Stage1.
1. Image -> preprocessing(patchify) -> encoder -> embeddings
2. user prompt -> tokenize -> embeddings
3. concat
=======

'''

from dataclasses import dataclass
import torch
from datasets import load_dataset

@dataclass
class Config:
    # Vision encoder (SigLIP, vision tower only)
    vision_name: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 196          # (224 / 16)^2 = 14 x 14
    vision_hidden: int = 768

    # Connector (pixel shuffle x2 + MLP)
    shuffle_factor: int = 2         # 196 -> 49 tokens, 768 -> 768*4 = 3072
    num_image_tokens: int = 49

    # LLM (SmolLM2) + its tokenizer
    lm_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lm_hidden: int = 576

    # For training (Stage 1: connector 만 학습, SigLIP / LLM 은 얼림)
    max_text_len: int = 96          # system + 질문 + 답변 토큰 (이미지 196 개는 별도)
    batch_size: int = 16
    lr: float = 1e-3                # connector 만 학습할 땐 크게 (LLaVA stage1 = 1e-3)
    warmup_steps: int = 100
    training_steps: int = 3000
    val_images: int = 1000          # 맨 뒤 이미지 1000 장을 val 로 (이미지 단위로 나눔)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class VLM(torch.nn.Module):
    '''
    main() 에서 한 흐름을 클래스로 묶은 것.
    [img x196 | text] -> SmolLM2 -> logits / loss
    '''
    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import SiglipVisionModel, AutoModelForCausalLM
        self.vision = SiglipVisionModel.from_pretrained(cfg.vision_name)
        self.lm = AutoModelForCausalLM.from_pretrained(cfg.lm_name, dtype=torch.float32)
        self.connector = torch.nn.Linear(cfg.vision_hidden, cfg.lm_hidden)  # 768 -> 576 (새로 학습하는 부분)

        # Stage 1: SigLIP, LLM 얼림 (requires_grad False). no_grad 는 안 씀 -> gradient 가 LLM 을 지나 connector 까지 흐름
        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.lm.parameters():
            # p.requires_grad = False     # Stage 1: LLM 얼림
            p.requires_grad = True        # Stage 2: LLM 풀기 (SigLIP 은 위에서 계속 얼림)

    def image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():                                               # vision 앞에는 학습할 게 없어서 no_grad OK
            h = self.vision(pixel_values=pixel_values).last_hidden_state    # [B, 196, 768]
        return self.connector(h)                                            # [B, 196, 576]

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        img = self.image_embeds(pixel_values)                               # [B, 196, 576]
        txt = self.lm.get_input_embeddings()(input_ids)                     # [B, L, 576]
        B, N = img.shape[:2]

        inputs_embeds = torch.cat([img, txt], dim=1)                        # [B, 196 + L, 576]
        attention_mask = torch.cat([attention_mask.new_ones(B, N), attention_mask], dim=1)   # 이미지는 항상 1
        if labels is not None:
            labels = torch.cat([labels.new_full((B, N), -100), labels], dim=1)               # 이미지는 채점 X
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, pixel_values, gen_ids, eos_id, max_new_tokens=20):
        '''pixel_values [1, 3, 224, 224], gen_ids [1, L_prompt] (assistant\\n 까지) -> 답변 token id list'''
        inputs_embeds = torch.cat([self.image_embeds(pixel_values), self.lm.get_input_embeddings()(gen_ids)], dim=1)
        out = self.lm(inputs_embeds=inputs_embeds, use_cache=True)          # prefill
        cache, answer = out.past_key_values, []
        for _ in range(max_new_tokens):                                     # decode (새 토큰 1 개 + cache)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() == eos_id:
                break
            answer.append(next_id.item())
            out = self.lm(input_ids=next_id, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        return answer


def main():
    cfg = Config()

    #DATASET.
    #https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
    DATASET_NAME = "HuggingFaceM4/the_cauldron"
    data = load_dataset(DATASET_NAME,"cocoqa", split="train", streaming=True)

    sample = next(iter(data))
    image = sample["images"][0]
    turn = sample["texts"][0]  # list[dict]: user, assistant, source


    # 1. Image(640,480) -> preprocessing(patchify) -> encoder -> embeddings
    from transformers import SiglipImageProcessor, SiglipVisionModel
    vision = SiglipVisionModel.from_pretrained(cfg.vision_name)
    processor = SiglipImageProcessor.from_pretrained(cfg.vision_name)
    
    #processor(images=image, return_tensors="pt") -> pixel_values: (1, 3, 224, 224)
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    out = vision(pixel_values=pixel_values)
    image_embeds = out.last_hidden_state
    print(image_embeds.shape) # (1, 196, 768)

    # 2. Text -> chat template -> token ids (Cauldron is not a raw string)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name)

    #make a message list for the further analyze.
    messages = [
        {"role": "user", "content": turn["user"]},
        {"role": "assistant", "content": turn["assistant"]},
    ]

    # Full conversation turn (train: mask user tokens in loss later)
    # messages(dict list) -> "<|im_start|>system ... user ... assistant ...<|im_end|>" 문장 -> token ids
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )["input_ids"]                                                              # [1, L]
    print("input_ids", tuple(input_ids.shape))
    print(tokenizer.decode(input_ids[0, :80], skip_special_tokens=False))

    # Inference-style: user only + empty assistant slot
    # add_generation_prompt=True -> 끝에 "<|im_start|>assistant\n" 까지만 붙여줌 (여기서부터 모델이 생성)
    gen_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": turn["user"]}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"]                                                              # [1, L_prompt]
    print("gen_prompt_ids", tuple(gen_ids.shape))

    # 3. labels: assistant 답변에만 loss (나머지는 -100) for the training.
    #    prompt 부분 = gen_ids (user + "<|im_start|>assistant\n") -> 그 길이만큼 앞을 가림
    #    pad = eos = <|im_end|>(2) 라서 pad 로 판단하면 안 됨 -> 길이로 자름
    labels = input_ids.clone()
    labels[:, : gen_ids.size(1)] = -100
    print("labels (loss O):", tokenizer.decode(labels[0][labels[0] != -100]))   # "White.<|im_end|>\n"

    # 4. now we need to convert text -> embeddings
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(cfg.lm_name, dtype=torch.float32)   # 기본 bf16 -> SigLIP/connector(fp32) 와 맞춤
    text_embeds = lm.get_input_embeddings()(input_ids)
    print("text_embeds", tuple(text_embeds.shape))


    # 5. Conncetor Image embeedign -> text embeeding
    image_embeds_dim = image_embeds.shape[-1]
    text_embeds_dim =text_embeds.shape[-1]
    from torch import nn
    connector = nn.Linear(image_embeds_dim, text_embeds_dim)
    image_embeds = connector(image_embeds)

    #6. Concatenate image and teezt embeddings
    # [B, L_image, D] + [B, L_text, D] -> [B, L_image + L_text, D]
    prefix = torch.cat([image_embeds, text_embeds], dim=1)
    print("prefix", tuple(prefix.shape))

    # =========================================================
    # 7. Inference (generate) with KV cache
    #    학습용 prefix 는 정답(assistant 답변)까지 들어있음 -> 추론은 gen_ids (assistant\n 까지) 로 다시 만듦
    #    [img x196 | <|im_start|>system ... user 질문 ... <|im_start|>assistant\n]  -> 여기서부터 생성
    # =========================================================
    with torch.no_grad():
        gen_embeds = lm.get_input_embeddings()(gen_ids)                     # [1, L_prompt, 576]
        gen_prefix = torch.cat([image_embeds, gen_embeds], dim=1)           # [1, 196 + L_prompt, 576]

        # 7-1. prefill: prompt 전체를 한 번에 넣고 cache 생성 (내부에서 causal mask 자동)
        out = lm(inputs_embeds=gen_prefix, use_cache=True)
        cache = out.past_key_values                                          # layer 30 개 x (K, V)
        print("prefill logits", tuple(out.logits.shape))           # [1, 196 + L_prompt, 49152]

        # 7-2. decode: 마지막 위치 예측 -> 새 토큰 1 개 + cache 로 다음 예측 (mask 불필요)
        answer = []
        for _ in range(20):
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)    # [1, 1]
            if next_id.item() == tokenizer.eos_token_id:                    # <|im_end|> 나오면 멈춤
                break
            answer.append(next_id.item())
            out = lm(input_ids=next_id, past_key_values=cache, use_cache=True)   # 새 토큰은 id 로 넣어도 됨 (embedding 자동)
            cache = out.past_key_values

    print("cache length", cache.get_seq_length())                          # 196 + L_prompt + 생성 토큰 수
    print("question:", turn["user"])
    print("answer  :", tokenizer.decode(answer))                            # connector 가 랜덤이라 엉뚱한 답이 정상
    print("gt      :", turn["assistant"])





if __name__ == "__main__":
    main()