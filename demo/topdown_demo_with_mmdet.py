# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Function to check if a point is inside a polygon
def point_in_quad(pt, quad):
    return cv2.pointPolygonTest(np.array(quad, dtype=np.int32), (int(pt[0]), int(pt[1])), False) >= 0

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # get masks associated with current bboxes
    masks = np.zeros((bboxes.shape[0], img.shape[0], img.shape[1]),
                        dtype=np.uint8)
    for i in range(bboxes.shape[0]):
        mask = pred_instance.masks[i]
        masks[i] = mask

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    # put the masks into img with transparency
    # Key Points are data_samples.key_points
        # Left Shoulder is 5
        # Right Shoulder is 6
        # Left Hip is 11
        # Right Hip is 12
    
    kp = data_samples.pred_instances.keypoints[0]

    # Assume kp, img, masks are defined as before
    p1 = kp[5]   # Left Shoulder
    p2 = kp[6]   # Right Shoulder
    p3 = kp[11]  # Left Hip
    p4 = kp[12]  # Right Hip

    h, w = img.shape[:2]

    # Get line equations: y = m*x + b
    def line_y(x, pt1, pt2):
        # Handle vertical lines
        if pt2[0] == pt1[0]:
            return np.full_like(x, min(pt1[1], pt2[1]))
        m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        b = pt1[1] - m * pt1[0]
        return m * x + b

    def get_perpendicular_line_y(x, pt1, pt2):
        # Handle vertical lines
        if pt2[0] == pt1[0]:
            return np.full_like(x, pt1[1])
        m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        m_perpendicular = -1 / m
        b = pt1[1] - m_perpendicular * pt1[0]
        return m_perpendicular, b, m_perpendicular * x + b

    x_coords = np.arange(w)

    # Function to compute perpendicular line through a point and reference segment
    def get_perpendicular_line_through(p_base, p_ref):
        dx = p_ref[0] - p_base[0]
        dy = p_ref[1] - p_base[1]

        if dx == 0:  # vertical segment, perpendicular is horizontal
            m_perp = 0
            b = p_base[1]
            y_vals = np.full_like(x_coords, b)
        elif dy == 0:  # horizontal segment, perpendicular is vertical (x constant)
            m_perp = np.inf
            b = p_base[0]
            y_vals = None  # special case
        else:
            m = dy / dx
            m_perp = -1 / m
            b = p_base[1] - m_perp * p_base[0]
            y_vals = m_perp * x_coords + b

        return y_vals, (m_perp, b)

    # Get perpendicular lines through p1→p3 and p2→p4
    y_left, (m_left, b_left) = get_perpendicular_line_through(p1, p3)
    y_right, (m_right, b_right) = get_perpendicular_line_through(p2, p4)

    mean_m = (m_left + m_right) / 2
    mean_top_point = (p1 + p2) / 2
    mean_bottom_point = (p3 + p4) / 2
    y_right = mean_m * x_coords + (mean_top_point[1] - mean_m * mean_top_point[0])
    y_left = mean_m * x_coords + (mean_bottom_point[1] - mean_m * mean_bottom_point[0])

    # Initialize drawing image
    img = img.copy()

    # Handle left side
    if y_left is None:
        y_l = np.arange(h)
        x_l = np.full_like(y_l, int(b_left))
    else:
        x_l = x_coords.astype(int)
        y_l = y_left.astype(int)

    # Handle right side
    if y_right is None:
        y_r = np.arange(h)
        x_r = np.full_like(y_r, int(b_right))
    else:
        x_r = x_coords.astype(int)
        y_r = y_right.astype(int)

    # Draw left perpendicular line (red)
    for x, y in zip(x_l, y_l):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # Draw right perpendicular line (green)
    for x, y in zip(x_r, y_r):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    # Optionally: draw the original body lines for reference
    cv2.line(img, tuple(p1.astype(int)), tuple(p3.astype(int)), (255, 0, 0), 2)  # Left (blue)
    cv2.line(img, tuple(p2.astype(int)), tuple(p4.astype(int)), (255, 255, 0), 2)  # Right (cyan)

    # Create a mask for all pixels between the lines
    Y, X = np.ogrid[:h, :w]
    between_mask = (Y >= y_right[X]) & (Y <= y_left[X])

    if masks is not None:
        for i in range(masks.shape[0]):
            mask = masks[i]  # shape: (H, W)
            # Keep only the part of the mask between the lines
            filtered_mask = mask & between_mask

            # Prepare for blending
            filtered_mask_expanded = np.expand_dims(filtered_mask, axis=-1)

            # Blend: apply transparency only inside the region
            img = np.where(
                filtered_mask_expanded == 0,
                img,
                img * 0.5 + 0.5 * 255
            ).astype(np.uint8)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        #'/home/diego/workspace/TestLab/mmpose/checkpoints/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py',#
        args.det_config, 
        #'/home/diego/workspace/TestLab/mmpose/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth',#
        args.det_checkpoint, 
        device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # topdown pose estimation
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
