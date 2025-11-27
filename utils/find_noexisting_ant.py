import json
import os
from tqdm import tqdm

anno = json.load(open("/YOUR_PATH/data/umt_data/anno_downstream/anet_ret_val.json"))

cnt = 0
for info in tqdm(anno):
    video = info['video']
    pth = os.path.join("/YOUR_PATH/data/umt_data/anet/ori_data/v1-3/test", video)
    if not os.path.exists(pth) or not os.path.exists(pth[:-4]+'.mkv'):
        cnt += 1
        print(pth)
print("total", len(anno))
print(cnt)