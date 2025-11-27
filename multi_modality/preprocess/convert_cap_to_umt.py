import json

# 使用 with 语句安全地读取文件
with open("/YOUR_PATH/data/umt_data/anno_downstream/cap/test_descs_cap.json", 'r') as f:
    cap_file = json.load(f)

anno_final = []
for item in cap_file:
    video = item['video_id']
    cap = item['desc']
    # 添加 .mp4 扩展名到视频 ID
    anno_final.append({"video": video + '.mp4', "caption": cap})

# 使用 with 语句安全地写入文件
with open("/YOUR_PATH/data/umt_data/anno_downstream/cap/msrvtt_cap_test.json", 'w') as f:
    json.dump(anno_final, f, indent=4)  # 添加 indent 使输出更易读