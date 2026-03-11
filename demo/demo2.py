import argparse
import tempfile
import numpy as np
import torch
import cv2
import mmcv
import mmengine

from mmengine import DictAction
from mmengine.utils import track_iter_progress
from mmengine.registry import DefaultScope

from mmaction.apis import detection_inference, pose_inference
from mmaction.apis import init_recognizer, inference_skeleton
from mmaction.utils import frame_extract

from mmpose.registry import VISUALIZERS

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

    parser = argparse.ArgumentParser(description="MMAction2 sliding window demo")

    parser.add_argument("video")
    parser.add_argument("out_filename")

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)

    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)

    parser.add_argument("--label-map", required=True)

    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--det-score-thr", type=float, default=0.9)
    parser.add_argument("--det-cat-id", type=int, default=0)
    parser.add_argument("--short-side", type=int, default=480)

    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=15)

    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})

    return parser.parse_args()


def sliding_window_inference(model, pose_results, h, w, window_size, stride):

    results = []

    num_frames = len(pose_results)

    for start in range(0, num_frames - window_size + 1, stride):

        end = start + window_size

        window_pose = pose_results[start:end]

        result = inference_skeleton(model, window_pose, (h, w))

        label = result.pred_score.argmax().item()
        score = result.pred_score.max().item()

        results.append({
            "start": start,
            "end": end,
            "label": label,
            "score": score
        })

    return results


def visualize(args, frames, datasamples, frame_labels):

    DefaultScope.get_instance("mmpose_visualizer", scope_name="mmpose")

    pose_cfg = mmengine.Config.fromfile(args.pose_config)

    visualizer = VISUALIZERS.build(pose_cfg.visualizer)

    visualizer.set_dataset_meta(datasamples[0].dataset_meta)

    vis_frames = []

    print("Drawing skeleton >>>>")

    for datasample, frame, label in track_iter_progress(
        list(zip(datasamples, frames, frame_labels))
    ):

        frame = mmcv.imconvert(frame, "bgr", "rgb")

        visualizer.add_datasample(
            name="result",
            image=frame,
            data_sample=datasample,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3
        )

        img = visualizer.get_image()

        cv2.putText(
            img,
            label,
            (10, 30),
            FontFace,
            FontScale,
            FontColor,
            Thickness,
            LineType
        )

        vis_frames.append(img)

    video = mpy.ImageSequenceClip(vis_frames, fps=24)

    video.write_videofile(args.out_filename, remove_temp=True)


def main():

    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()

    frame_paths, frames = frame_extract(
        args.video,
        args.short_side,
        tmp_dir.name
    )

    h, w, _ = frames[0].shape


    print("Processing: Human Detection")

    det_results, _ = detection_inference(
        args.det_config,
        args.det_checkpoint,
        frame_paths,
        args.det_score_thr,
        args.det_cat_id,
        args.device
    )


    print("Processing: Pose Estimation")

    pose_results, pose_data_samples = pose_inference(
        args.pose_config,
        args.pose_checkpoint,
        frame_paths,
        det_results,
        args.device
    )

    torch.cuda.empty_cache()


    print("Loading Action Model")

    cfg = mmengine.Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(cfg, args.checkpoint, args.device)


    print("Sliding Window Inference")

    window_results = sliding_window_inference(
        model,
        pose_results,
        h,
        w,
        args.window_size,
        args.stride
    )


    label_map = [x.strip() for x in open(args.label_map)]

    frame_labels = [""] * len(frames)


    for r in window_results:

        label = label_map[r["label"]]

        for i in range(r["start"], r["end"]):

            frame_labels[i] = label


    visualize(args, frames, pose_data_samples, frame_labels)

    tmp_dir.cleanup()

    print("DONE")


if __name__ == "__main__":
    main()