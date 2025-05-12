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
    back_curvature = back_area / back_perimeter
    return back_curvature

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
    #print(data_samples.pred_instances)

    kp = data_samples.pred_instances.keypoints[0]

    vis = data_samples.pred_instances.keypoints_visible[0]

    # Assume kp, img, masks are defined as before
    p1, v1 = kp[5], vis[5]  # Left Shoulder
    nose, nos_vis = kp[0], vis[0]  # Nose
    left_eye, left_eye_vis = kp[1], vis[1]  # Left Eye
    right_eye, right_eye_vis = kp[2], vis[2]  # Right Eye
    p2, v2 = kp[6], vis[6]  # Right Shoulder
    p3, v3 = kp[11], vis[11]  # Left Hip
    p4, v4 = kp[12], vis[12]  # Right Hip

    if v1 <= 0.75 and v3 <= 0.75:
        view = 'right'
    elif v2 <= 0.75 and v4 <= 0.75:
        view = 'left'
    else:
        view = 'front'
        
    h, w = img.shape[:2]

    x_coords = np.arange(w)

    if view == 'right':
        y_right, (m_right, b_right) = get_perpendicular_line_through(x_coords, p2, p4)
        y_top = m_right * x_coords + (p2[1] - m_right * p2[0])
        y_bottom = m_right * x_coords + (p4[1] - m_right * p4[0])
        # line between p2 and p4
        m_vertical = (p4[1] - p2[1]) / (p4[0] - p2[0])
        b_vertical = p2[1] - m_vertical * p2[0]
        y_vertical = m_vertical * x_coords + b_vertical
    elif view == 'left':
        y_left, (m_left, b_left) = get_perpendicular_line_through(x_coords, p1, p3)
        y_top = m_left * x_coords + (p1[1] - m_left * p1[0])
        y_bottom = m_left * x_coords + (p3[1] - m_left * p3[0])
        # line between p1 and p3
        m_vertical = (p3[1] - p1[1]) / (p3[0] - p1[0])
        b_vertical = p1[1] - m_vertical * p1[0]
        y_vertical = m_vertical * x_coords + b_vertical
    else:
        # Get perpendicular lines through p1→p3 and p2→p4
        y_left, (m_left, b_left) = get_perpendicular_line_through(x_coords, p1, p3)
        y_right, (m_right, b_right) = get_perpendicular_line_through(x_coords, p2, p4)

        mean_m = (m_left + m_right) / 2
        mean_top_point = (p1 + p2) / 2
        mean_bottom_point = (p3 + p4) / 2
        # line between p1 and p3
        m_vertical = (p3[1] - p1[1]) / (p3[0] - p1[0])
        b_vertical = p1[1] - m_vertical * p1[0]
        y_vertical = m_vertical * x_coords + b_vertical
        y_top = mean_m * x_coords + (mean_top_point[1] - mean_m * mean_top_point[0])
        y_bottom = mean_m * x_coords + (mean_bottom_point[1] - mean_m * mean_bottom_point[0])

    # Create a mask for all pixels between the lines
    Y, X = np.ogrid[:h, :w]
    between_mask = (Y >= y_top[X]) & (Y <= y_bottom[X]) 

    if view != 'front':
        # Check if nose point is to the left or right of the vertical line
        if view == 'left':
            eye_side = 'right' if int(left_eye[0]) > y_vertical[int(left_eye[0])] else 'left'   
        elif view == 'right':
            eye_side = 'right' if int(right_eye[0]) > y_vertical[int(right_eye[0])] else 'left'

        # Select only the part of the mask on the opposite side of the nose respect to the vertical line
        # So need to filter based on y_vertical
        if eye_side == 'right':
            between_mask = between_mask & (Y <= y_vertical[X])
        else:
            between_mask = between_mask & (Y >= y_vertical[X])

    if masks is not None:
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
                        # Find the points corresponding to p2 and p6:
                        shoulder = np.argmin(np.linalg.norm(approx - p2, axis=2))
                        hip = np.argmin(np.linalg.norm(approx - p4, axis=2))
                    elif view == 'left':
                        shoulder = np.argmin(np.linalg.norm(approx - p1, axis=2))
                        hip = np.argmin(np.linalg.norm(approx - p3, axis=2))

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
                        #print(back_curvature)   
                        # Show back curvature as text in img
                        cv2.putText(img, f'Back Curvature: {back_curvature:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        # Plot back points as points in img
                        for point in back_points:
                            cv2.circle(img, tuple(point), 2, (255, 0, 255), 2)

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
