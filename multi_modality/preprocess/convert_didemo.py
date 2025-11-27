import os
import subprocess
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

input_dir = "/YOUR_PATH/data/umt_data/didemo_ori/train_videos"
output_dir = "/YOUR_PATH/data/umt_data/didemo_ori/train_videos_mp4"
output_suffix = ".mp4"
log_failed = "/tmp/failed_videos.txt"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

def is_valid_video(path):
    """使用 ffprobe 检查视频是否有效"""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and result.stdout.strip() != b''

def convert_to_mp4(filename):
    input_path = os.path.join(input_dir, filename)
    if not os.path.isfile(input_path):
        return

    base, ext = os.path.splitext(filename)
    real_input = input_path
    is_temp = False

    if ext == "":
        temp_input = input_path + ".mpg"
        try:
            shutil.copy(input_path, temp_input)
        except Exception as e:
            print(f"❌ Failed to copy {input_path}: {e}")
            return
        real_input = temp_input
        is_temp = True

    if not is_valid_video(real_input):
        print(f"🚫 Invalid video, skipping: {input_path}")
        if is_temp and os.path.exists(real_input):
            os.remove(real_input)
        return

    output_path = os.path.join(output_dir, base + output_suffix)

    print(f"🔄 Converting {real_input} -> {output_path}")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", real_input, 
        "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-an", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 100:
        print(f"✅ Converted: {output_path}")
    else:
        print(f"❌ Failed: {input_path}")
        with open(log_failed, "a") as f:
            f.write(input_path + "\n")
        if os.path.exists(output_path):
            os.remove(output_path)

    if is_temp and os.path.exists(real_input):
        os.remove(real_input)

if __name__ == "__main__":
    filenames = [
        # "10279741@N00_6757697855_6f83d834a2.wmv",
        # "86261473@N00_5189746556_2883a64df0.avi",
        # "56106041@N00_3688563265_3f0239fd9d.mp4",
        # "33195950@N00_2745600175_65f14299f0.mp4",
        # "10279741@N00_8437050600_da41f17538.wmv",
        # "37727860@N00_2587724820_f23fee6fae.mov",
        # "10279741@N00_8227850332_625c49afa5.wmv",
        # "15486718@N00_2989437659_76bd82e8e4.mp4",
        "35468136000@N01_2802727517_73a175c654.mov",
        # "14587494@N00_6183405897_212ddd5f7a.wmv",
        # "38037391@N02_5506896844_68e5888e94.m4v",
        # "62976015@N00_3419554494_6acd3b9264.mp4",
        # "43255570@N00_3409262841_685835bc20.wmv",
        # "39326559@N00_6229514785_158d0219f5.mp4"
        ]
    # filenames = os.listdir(input_dir)
    num_workers = min(cpu_count(), 24)  # 最多使用8核，避免磁盘IO过载
    print(f"🌟 Starting conversion with {num_workers} workers...")

    with Pool(num_workers) as pool:
        pool.map(convert_to_mp4, filenames)

    print("🎉 All done.")
