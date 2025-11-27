import json
import os

anno = json.load(open("/YOUR_PATH/data/umt_data/anno_downstream/didemo_ret_train.json"))
path = "/YOUR_PATH/data/umt_data/didemo_ori/train_videos_mp4"
all_video_in_pth = os.listdir(path)

video_list = [item.split('.')[0] for item in all_video_in_pth]
# import pdb;pdb.set_trace()
cnt = 0
for item in anno:
    vdo_pth = item['video'].split('.')[0]
    if vdo_pth not in video_list:
        print(vdo_pth)
        cnt += 1
print(cnt)
print(len(anno))