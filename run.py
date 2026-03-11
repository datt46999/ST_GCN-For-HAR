import os

os.makedirs("/demo/results", exist_ok=True)

video_path = "/workspace/data/3.mp4"
output_video = "/workspace/data/output_video3.mp4"

action_config = "/workspace/STGCN_mmaction/configs/config.py"
action_checkpoint = "./work_dirs/config/epoch_16.pth"

det_config = "/workspace/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_giou_1x_coco.py"
det_checkpoint = "/workspace/data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

pose_config = "/workspace/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
pose_checkpoint = "/workspace/data/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"

label_map = "/workspace/data/label_map_ntu60.txt"
# \\d:\Graph\MMAction2\mmpose
cmd = f"""
python demo/demo.py \
{os.path.abspath(video_path)} \
{os.path.abspath(output_video)} \
--config {os.path.abspath(action_config)} \
--checkpoint {os.path.abspath(action_checkpoint)} \
--det-config {det_config} \
--det-checkpoint {det_checkpoint} \
--pose-config {pose_config} \
--pose-checkpoint {pose_checkpoint} \
--label-map {label_map}
"""

print(cmd)
ret = os.system(cmd)
print("Return code:", ret)
print("Output saved to:", output_video)
