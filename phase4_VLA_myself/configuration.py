"""
configuration.py — phase4 VLA 설계도 (A 방식: layer 별 KV 공유)

prefix [img x2 frame | lang | state] -> SmolLM2 (VLM)
suffix [noisy action chunk + t]      -> action expert
Target: PushT (state 2D, action 2D, image 96x96, 10fps)
"""
from dataclasses import dataclass
import torch
@dataclass
class Config:

    #device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # input / output (PushT)
    state_dim: int = 2              # agent (x, y)
    action_dim: int = 2             # action (x, y)
    fps: int = 10
    chunk_size: int = 16            # action chunk (1.6 초)
    n_frames: int = 2               # 현재 + 0.1 초 전

    # Vision encoder (SigLIP vision tower, phase3 와 같음)
    vision_name: str = "google/siglip-base-patch16-224"
    image_size: int = 224           # 96 -> 224 resize
    num_patches: int = 196          # (224 / 16)^2, frame 당 img 토큰 수
    vision_hidden: int = 768

    # LLM = VLM 의 "decoder" 쪽 (SmolLM2, phase3 와 같음)
    lm_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lm_hidden: int = 576            # 모든 prefix 토큰이 맞춰야 하는 차원
    lm_layers: int = 12
    
    # max_lang_len: int = 32          # task 문장 토큰 padding 길이
    max_lang_len: int = 20          # PushT 최대 13 토큰 + 여유 7 (pad)
    expert_layers: int = 12

    # Action expert (layer 수 / kv 크기는 LLM 과 같아야 K, V 를 이어 붙일 수 있음)
    '''
    	VLM (Gemma 2B)	Expert (Gemma 300M)
        width (hidden)	2048	1024 ← 줄임
        depth (layer)	18	18
        q head / kv head	8 / 1	8 / 1 ← 같음
        head_dim	256	256 ← 같음
    '''
    expert_hidden: int = 384
    head_dim: int = 64
    num_q_heads: int = 9
    num_kv_heads: int = 3      
    expert_ff_hidden: int = 1536

    #Groupd Multi Head attentio dimensions
    kv_hidden: int = head_dim * num_kv_heads # kv dimension
    Group_size: int = num_q_heads // num_kv_heads

    # flow matching
    num_denoise_steps: int = 10 
    T: float = 1.0
    #Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 100               # (안 씀: step 단위로 학습 -> training_steps)

    # Training (step 단위, phase3 와 같은 방식)
    training_steps: int = 30000     # phase2 는 5,300 step 에서 학습 부족 -> 더 길게
    warmup_steps: int = 500
    lr_new: float = 1e-4            # 새로 만든 것: expert, projector (랜덤 시작 -> 크게)
    lr_pretrained: float = 1e-5     # SigLIP, SmolLM2 (이미 학습됨 -> 작게, phase3 Stage 2 처럼)
    grad_clip: float = 1.0
    num_workers: int = 8            # mp4 디코딩이 CPU 일 -> worker 여러 개
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 20           # val 은 고정된 20 batch 로 매번 같은 문제 채점