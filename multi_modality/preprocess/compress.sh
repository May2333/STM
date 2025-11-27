fps=2
size=224
file_type=video
input_root=/YOUR_PATH/data/umt_data/didemo_ori/test_videos
input_file_list_path=/YOUR_PATH/data/umt_data/didemo_list.txt
# you may use `ls -U ${input_root} > ${input_file_list_path}` to efficiently generate the file above.
output_root=/YOUR_PATH/data/umt_data/processed_didemo_videos/test_videos
python compress.py \
--input_root=${input_root} --output_root=${output_root} \
--input_file_list_path=${input_file_list_path} \
--fps=${fps} --size=${size} --file_type=${file_type} --num_workers 24 