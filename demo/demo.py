import argparse
import os
import tempfile

import torch
import cv2
import mmcv
import mmengine

import mmpose.visualization
from mmengine import DictAction
from mmengine.utils import track_iter_progress
from mmengine.registry import DefaultScope

from mmaction.apis import detection_inference, pose_inference, init_recognizer, inference_skeleton
from mmaction.utils import frame_extract
from mmpose.registry import VISUALIZERS
import numpy as np
try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError("Please install moviepy to enable output file")


FontFace = cv2.FONT_HERSHEY_DUPLEX
FontScale = 0.75
FontColor = (255, 255, 255)
Thickness = 1
LineType = 1

def parse_args():
    """
    all functions was config:
    input_video, out_filename, 
    config (action_config), checkpoint (action_checkpoint),
    cfg_det (detector config), det_checkpoint,
    cfg_pose, pose_checkpoint,
    device, label_map,
    det_score_thr, det_cat_id, short_side, cfg_options
    """
    parser = argparse.ArgumentParser(description="MMAction2 skeleton demo")
    parser.add_argument("video", help = "input_video")
    parser.add_argument("out_filename", help = "output video")

    parser.add_argument("--config", help ="action config", required= True)
    parser.add_argument('--checkpoint', help='action checkpoint', required=True)

    parser.add_argument('--det-config', help='detector config', required=True)
    parser.add_argument('--det-checkpoint', help='detector checkpoint', required=True)
    parser.add_argument('--pose-config', help='pose config',required=True)
    parser.add_argument('--pose-checkpoint', help='pose checkpoint', required=True)

    parser.add_argument('--label-map', required=True)

    parser.add_argument('--device', default = "cuda:0")
    parser.add_argument('--det-score-thr', type = float, default= 0.9 )
    parser.add_argument('--det-cat-id', type = int, default= 0)
    parser.add_argument('--short-side',type = int, default= 480)
    parser.add_argument('--cfg-options', nargs = '+', action = DictAction, default={})
    return parser.parse_args()


def visualize(args, frames, datasamples, action_labels):
    """
    get scope to mmpose 
    visualize with earch datasample and frame of input 

    """

    DefaultScope.get_instance(
        "mmpose_visualizer",
        scope_name = "mmpose"
    )

    pose_cfg = mmengine.Config.fromfile(args.pose_config)
    visualizer = VISUALIZERS.build(pose_cfg.visualizer)
    visualizer.set_dataset_meta(datasamples[0].dataset_meta)

    vis_frame = []
    print("Drawing skeleton >>>>")

    for datasample, frame in track_iter_progress(list(zip(datasamples, frames))):
        frame = mmcv.imconvert(frame, 'bgr', 'rgb')
        visualizer.add_datasample(
            name = 'results',
            image = frame,
            data_sample = datasample,
            draw_gt = False, draw_heatmap = False,
            draw_bbox = True, show = False,
            wait_time = 0, out_file = None, kpt_thr = 0.3
        )
        img = visualizer.get_image()
        cv2.putText(
            img, action_labels,
            (10, 30),
            FontFace, FontScale, FontColor, Thickness,
            LineType
        )
        vis_frame.append(img)
    video = mpy.ImageSequenceClip(vis_frame, fps = 24)
    video.write_videofile(args.out_filename, remove_temp = True)
def main():
    """
    all steps
    human detect -> pose estimate -> action recognition
    """
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, frames = frame_extract(
        args.video, args.short_side, tmp_dir.name
    )
    h, w, _ = frames[0].shape

    # human detection
    print("Processing: detect Human........")
    det_results,_= detection_inference(
        args.det_config,
        args.det_checkpoint,
        frame_paths,
        args.det_score_thr,
        args.det_cat_id,
        args.device
    )

    # print("det_results example:", det_results.shape)
    torch.cuda.empty_cache()
    print("Processing: Pose estimation ........")
    # pose estimate 
    pose_results, pose_data_samples= pose_inference(
        args.pose_config,
        args.pose_checkpoint,
        frame_paths, 
        det_results,
        args.device
    )
    torch.cuda.empty_cache()
    # human action recognition
    print("Processing: Predict ........")
    cfg = mmengine.Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(cfg, args.checkpoint, args.device)
    result = inference_skeleton(model, pose_results, (h, w))
    label_map = [x.strip() for x in open(args.label_map)]
    action_labels = label_map[result.pred_score.argmax().item()]
    visualize(args, frames, pose_data_samples,action_labels)
    tmp_dir.cleanup()
if __name__ == "__main__":
    main()
    print("DONE :)")