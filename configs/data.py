import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

data_dir = '/YOUR_PATH/data/umt_data/'
if data_dir is None:
    raise ValueError("please set environment `VL_DATA_DIR` before continue")

data_root = __os.path.join(data_dir, "videos_images")
anno_root_pt = __os.path.join(data_dir, "anno_pretrain")
anno_root_downstream = __os.path.join(data_dir, "anno_downstream")

# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining datasets
    cc3m=[
        f"{anno_root_pt}/cc3m_emp.json", 
        "/YOUR_PATH/data/umt_data/cc3m_emp_processed_imgs"
    ],
    cc12m=[
        f"{anno_root_pt}/cc12m_train.json", 
        "your_cc12m_path"
    ],
    sbu=[
        f"{anno_root_pt}/sbu.json", 
        "your_sbu_path"
    ],
    vg=[
        f"{anno_root_pt}/vg.json", 
        "your_vg_path"
    ],
    coco=[
        f"{anno_root_pt}/coco.json", 
        "your_coco_path"
    ],
    webvid=[
        f"{anno_root_pt}/webvid_train.json", 
        "/YOUR_PATH/data/umt_data/processed_webvid_videos/",
        "video"
    ],
    webvid_10m=[
        f"{anno_root_pt}/webvid_10m_train.json",
        "your_webvid10m_path",
        "video",
    ],
    # downstream datasets.
)

# composed datasets.
# available_corpus["data_5m"] = [
#     available_corpus["webvid"], 
#     available_corpus["cc3m"]
# ]
available_corpus["data_5m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"]
]
available_corpus["data_17m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["data_25m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]

# ============== for validation =================
available_corpus["msrvtt_1k_test"] = [
    f"{anno_root_downstream}/msrvtt_test1k.json",
    "/YOUR_PATH/data/umt_data/MSRVTT/videos/all",
    "video",
]

