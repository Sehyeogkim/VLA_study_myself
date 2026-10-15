'''
Code when we jus tstart the project

1. check the data.
2. check the model 
'''

import torch
from datasets import load_dataset

#DATASET.
#https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
DATASET_NAME = "HuggingFaceM4/the_cauldron"
data = load_dataset(DATASET_NAME,"cocoqa", split="train", streaming=True)

for sample in data:
    print(sample["texts"])
    print(sample["images"])
    img = sample["images"][0]
    print(img.size) # (640,480)
    break

'''

1. image -> encoder -> token -> embeddings
2. use prompt -> token -> embeddings

prefix = {image_emb | text_emb}



'''