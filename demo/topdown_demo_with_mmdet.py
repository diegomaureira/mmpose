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

# Function to compute perpendicular line through a point and reference segment
def get_perpendicular_line_through(x_coords, p_base, p_ref):
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

def calculate_curvature(back_points):
    # Calculate the area of the back
    back_area = cv2.contourArea(back_points)
    # Calculate the perimeter of the back
    back_perimeter = cv2.arcLength(back_points, True)
    # Calculate the curvature: Area / Perimeter^2
    #back_curvature = back_area / back_perimeter
    # Compactness
    compactness = 4 * np.pi * back_area / (back_perimeter ** 2)
    return compactness

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
    selected_indexes = np.where(np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr))[0]
    
    bboxes = bboxes[selected_indexes]
    masks = pred_instance.masks[selected_indexes]

    masks = masks[nms(bboxes, args.nms_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    kp = data_samples.pred_instances.keypoints[0]
    vis = data_samples.pred_instances.keypoints_visible[0]

    # Assume kp, img, masks are defined as before
    nose, nose_vis = kp[0], vis[0]  # Nose
    left_eye, left_eye_vis = kp[1], vis[1]  # Left Eye
    right_eye, right_eye_vis = kp[2], vis[2]  # Right Eye
    left_ear, left_ear_vis = kp[3], vis[3]  # Left Ear
    right_ear, right_ear_vis = kp[4], vis[4]  # Right Ear
    left_shoulder, left_shoulder_vis = kp[5], vis[5]  # Left Shoulder
    right_shoulder, right_shoulder_vis = kp[6], vis[6]  # Right Shoulder
    left_elbow, left_elbow_vis = kp[7], vis[7]  # Left Elbow
    right_elbow, right_elbow_vis = kp[8], vis[8]  # Right Elbow
    left_wrist, left_wrist_vis = kp[9], vis[9]  # Left Wrist
    right_wrist, right_wrist_vis = kp[10], vis[10]  # Right Wrist
    left_hip, left_hip_vis = kp[11], vis[11]  # Left Hip
    right_hip, right_hip_vis = kp[12], vis[12]  # Right Hip
    left_knee, left_knee_vis = kp[13], vis[13]  # Left Knee
    right_knee, right_knee_vis = kp[14], vis[14]  # Right Knee
    left_ankle, left_ankle_vis = kp[15], vis[15]  # Left Ankle
    right_ankle, right_ankle_vis = kp[16], vis[16]  # Right Ankle
    
    # Calculate mean visibility of left and right side
    left_side_vis = np.mean([left_eye_vis, left_ear_vis, left_shoulder_vis, left_elbow_vis, left_wrist_vis, left_hip_vis, left_knee_vis, left_ankle_vis])
    right_side_vis = np.mean([right_eye_vis, right_ear_vis, right_shoulder_vis, right_elbow_vis, right_wrist_vis, right_hip_vis, right_knee_vis, right_ankle_vis])

    view_threshold = 0.05

    # Determine view based on visibility
    if (left_side_vis - right_side_vis) > view_threshold:
        view = 'left'
    elif (right_side_vis - left_side_vis) > view_threshold:
        view = 'right'
    else:
        view = 'front'
        
    h, w = img.shape[:2]

    x_coords = np.arange(w)

    if masks is not None:

        if view == 'right':
            # Get perpendicular lines through right_shoulder and right_hip
            _, (m_right, _) = get_perpendicular_line_through(x_coords, right_shoulder, right_hip)
            # Get the line at the top of the back
            y_top = m_right * x_coords + (right_shoulder[1] - m_right * right_shoulder[0])
            # Get the line at the bottom of the back
            y_bottom = m_right * x_coords + (right_hip[1] - m_right * right_hip[0])
            # Get the line between right_shoulder and right_hip
            m_vertical = (right_hip[1] - right_shoulder[1]) / (right_hip[0] - right_shoulder[0])
            b_vertical = right_shoulder[1] - m_vertical * right_shoulder[0]
            y_vertical = m_vertical * x_coords + b_vertical
        elif view == 'left':
            _, (m_left, _) = get_perpendicular_line_through(x_coords, left_shoulder, right_shoulder)
            # Get the line at the top of the back
            y_top = m_left * x_coords + (left_shoulder[1] - m_left * left_shoulder[0])
            # Get the line at the bottom of the back
            y_bottom = m_left * x_coords + (right_shoulder[1] - m_left * right_shoulder[0])
            # Get the line between left_shoulder and left_hip
            m_vertical = (left_hip[1] - left_shoulder[1]) / (left_hip[0] - left_shoulder[0])
            b_vertical = left_shoulder[1] - m_vertical * left_shoulder[0]
            y_vertical = m_vertical * x_coords + b_vertical
        else:
            # Get perpendicular lines through left_shoulder→right_shoulder and right_shoulder→right_hip
            _, (m_left, _) = get_perpendicular_line_through(x_coords, left_shoulder, right_shoulder)
            _, (m_right, _) = get_perpendicular_line_through(x_coords, right_shoulder, right_hip)
            # Get the mean of the two lines
            mean_m = (m_left + m_right) / 2
            mean_top_point = (left_shoulder + right_shoulder) / 2
            mean_bottom_point = (right_shoulder + right_hip) / 2
            # Get the line at the top of the back
            y_top = mean_m * x_coords + (mean_top_point[1] - mean_m * mean_top_point[0])
            # Get the line at the bottom of the back
            y_bottom = mean_m * x_coords + (mean_bottom_point[1] - mean_m * mean_bottom_point[0])
            # Get the line between mean_top_point and mean_bottom_point
            m_vertical = (mean_bottom_point[1] - mean_top_point[1]) / (mean_bottom_point[0] - mean_top_point[0])
            b_vertical = mean_top_point[1] - m_vertical * mean_top_point[0]
            y_vertical = m_vertical * x_coords + b_vertical

        # Create a mask for all pixels between the lines
        Y, X = np.ogrid[:h, :w]
        # If view is right, keep only the part of the mask between the lines and to the right of the vertical line
        # If view is left, keep only the part of the mask between the lines and to the left of the vertical line
        if view == 'right':
            between_mask = (Y >= y_top[X]) & (Y <= y_bottom[X]) & (Y <= y_vertical[X])
        elif view == 'left':
            between_mask = (Y >= y_top[X]) & (Y <= y_bottom[X]) & (Y >= y_vertical[X])
        else:
            between_mask = (Y >= y_top[X]) & (Y <= y_bottom[X])

        for i in range(masks.shape[0]):
            mask = masks[i]  # shape: (H, W)
            # Keep only the part of the mask between the lines
            filtered_mask = mask & between_mask
        
            # Find contours
            contours, _ = cv2.findContours(filtered_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get bigger contour and plot over img in purple
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
            
                # Aprox contour with a polygon of 4 points
                epsilon = 0.05 * cv2.arcLength(max_contour, True)
                approx = cv2.approxPolyDP(max_contour, epsilon, True)

                if len(approx) == 4:
                    if view == 'right':
                        shoulder = np.argmin(np.linalg.norm(approx - right_shoulder, axis=2))
                        hip = np.argmin(np.linalg.norm(approx - right_hip, axis=2))
                    elif view == 'left':
                        shoulder = np.argmin(np.linalg.norm(approx - left_shoulder, axis=2))
                        hip = np.argmin(np.linalg.norm(approx - right_shoulder, axis=2))

                    if view != 'front':
                        other_indices = np.delete(np.arange(4), [shoulder, hip])
                        back_top = approx[other_indices[0]][0]
                        back_bottom = approx[other_indices[1]][0]
                        # Find points on max_contour between back_top and back_bottom
                        contour_points = max_contour[:, 0, :]  # shape (N, 2)  
                        # Get from max_contour the points between back_top and back_bottom
                        back_bottom_arg = np.argmin(np.linalg.norm(contour_points - back_bottom, axis=1))
                        back_top_arg = np.argmin(np.linalg.norm(contour_points - back_top, axis=1))
                        if back_bottom_arg > back_top_arg:
                            back_points = contour_points[back_top_arg: back_bottom_arg+1]
                        else:
                            back_points = np.concatenate((contour_points[back_top_arg:], contour_points[:back_bottom_arg+1]))
                        # Calculate curvature of back_top and back_bottom
                        back_curvature = calculate_curvature(back_points)  
                        # Show back curvature as text in img
                        cv2.putText(img, f'Back Curvature: {back_curvature:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        # Select color based on curvature
                        if back_curvature > 5:
                            color = (255, 0, 0)
                        else:
                            color = (0, 255, 0)
                        # Plot points in img
                        for point in back_points:
                            cv2.circle(img, tuple(point), 2, color, 2)

            # Prepare for blending
            #filtered_mask_expanded = np.expand_dims(filtered_mask, axis=-1)

            # Blend: apply transparency only inside the region
            #img = np.where(
            #    filtered_mask_expanded == 0,
            #    img,
            #    img * 0.5 + 0.5 * 255
            #).astype(np.uint8)

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
            output_file += '.mright_hip'

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
            
            #break

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
                    fourcc = cv2.VideoWriter_fourcc(*'mright_hipv')
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
