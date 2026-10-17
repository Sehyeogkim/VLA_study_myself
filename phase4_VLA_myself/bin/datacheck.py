'''
check the schema of the data we are going to use.
'''
# from datasets import load_dataset

# data = load_dataset("lerobot/pusht", split="train", streaming = True)

# for i in data:
#     print(i)
#     '''
#     {'observation.state': [222.0, 97.0],
#     'action': [233.0, 71.0],
#     'episode_index': 0,
#     'frame_index': 0,
#     'timestamp': 0.0,
#     'next.reward': 0.19029748439788818,
#     'next.done': False,
#     'next.success': False,
#     'index': 0,
#     'task_index': 0}
#     '''
#     break

# load_dataset 은 parquet (숫자 표) 만 읽음 -> 이미지 (mp4) / task 문장 없음
# LeRobotDataset: parquet + mp4 디코딩 + task 문장까지 합쳐서 한 샘플로 줌
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]          # VLA_study_myself/
DATA_DIR = ROOT / "data" / "pusht"                  # 이미 받아둔 로컬 데이터 (다운로드 X)
ds = LeRobotDataset("lerobot/pusht", root=DATA_DIR)

print(ds)
# LeRobotDataset({
#     Repository ID: 'lerobot/pusht',
#     Number of selected episodes: '206',
#     Number of selected samples: '25650',
#     Features: '['observation.image', 'observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index']',
# })

#전부다 읽어서 데이터 확인.
print("="*50)
print("overall length of the dataset: ", len(ds)) # 25650
print("="*50)

for data in ds:
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"{k}: {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"{k}: {type(v).__name__}  {v}")
    break


print("="*50)

'''
observation.image: (3, 96, 96)  torch.float32
observation.state: (2,)  torch.float32
action: (2,)  torch.float32
episode_index: ()  torch.int64
frame_index: ()  torch.int64
timestamp: ()  torch.float32
next.reward: ()  torch.float32
next.done: ()  torch.bool
next.success: ()  torch.bool
index: ()  torch.int64
task_index: ()  torch.int64
task: str  Push the T-shaped block onto the T-shaped target.
'''

#check out the longest tasks token
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
longest_task = ""
for data in ds:
    task = data["task"]
    if len(tokenizer(task, return_tensors="pt").input_ids[0]) > len(tokenizer.encode(longest_task)):
        longest_task = task
print("longest task: ", longest_task)
print("length of the longest task: ", len(tokenizer.encode(longest_task)))
