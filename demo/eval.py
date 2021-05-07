from argparse import ArgumentParser
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from PIL import Image
import os, sys, time, logging
import cv2
from crop_sliding_window import crop_sliding_window, non_max_suppression_slow, csv_to_json
from logger import get_logger

logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)

num_to_category_dict = {0: 'bridge', 1: 'appearance_less', 2: 'excess_solder', 3: 'appearance'}
category_to_color_dict = {'bridge': [0, 0, 255], 'appearance_less': [255,191,0], 'excess_solder': [221,160,221], 'appearance': [0,165,255]}
default_color = [0, 255, 0]

def parse_out_and_filter_score(results, score_thr):
    batch_bboxes, batch_labels =[], []
    for ind, result in enumerate(results):
        #print('ind: ',ind)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr#score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
        batch_bboxes.append(bboxes)
        batch_labels.append(labels)
    return batch_bboxes, batch_labels

def predictor(model, crop_image_list, batch, score):
    bboxes_list, labels_list = list(), list()
    img_bs = [crop_image_list[i:i+batch] for i in range(0, len(crop_image_list), batch)]

    for imgs in img_bs:
        result = inference_detector(model, imgs)
        #result = inference_detector(model, im)
        #print('@@result: ',result)
        bboxes, labels = parse_out_and_filter_score(result, score)
        bboxes_list.extend(bboxes)
        labels_list.extend(labels)
    #print('labels_list', labels_list)
    #print('bboxes_list', bboxes_list)
    return bboxes_list, labels_list

def unify_batch_predictor_output(image_file_name, image_shape, crop_rect_list, bboxs, labels):
    image_h, image_w, image_c = image_shape
    # bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...]
    bbox_list = list()
    pred_boxes_list = list()
    scores_list = list()
    pred_classes_list = list()
    #inst = Instances((image_h, image_w))
    #print('crop_rect_list',crop_rect_list)
    for crop_rect, pred_boxes, pred_classes in zip(crop_rect_list, bboxs, labels):
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_rect
        #print('crop_rect, output',crop_rect, output)
        #output_cpu = output["instances"].to("cpu")
        #pred_boxes = output_cpu.pred_boxes
        #scores = output_cpu.scores
        #pred_classes = output_cpu.pred_classes.tolist()
        if not (len(pred_boxes)>0 and len(pred_classes)>0):continue
        #print('output:', pred_boxes, pred_classes)
        #pred_boxes, pred_classes = output
        #print('pred_boxes, pred_classes',pred_boxes, pred_classes)
        for pred_box, pred_class in zip(pred_boxes, pred_classes):
            #if not (len(pred_box)>0 and len(pred_class)>0):continue
            #print('pred_box, pred_class',pred_box, pred_class)

            #for box, cls in zip(pred_box, pred_class):
            xmin,ymin,xmax,ymax,error_score = pred_box
            global_xmin = int(crop_xmin+xmin); global_ymin = int(crop_ymin+ymin)
            global_xmax = int(crop_xmin+xmax); global_ymax = int(crop_ymin+ymax)
            w = xmax-xmin; h = ymax-ymin
            error_type = num_to_category_dict[pred_class]
            bbox_list.append([image_file_name, error_type, global_xmin, global_ymin, global_xmax, global_ymax, error_score])
            """
            print(global_xmin,global_xmax,global_ymin)
            # crop image coordinate
            bbox_crop_coord = [int(pred) for pred in pred_box[0:4].tolist()]
            # original image coordinate
            bbox_global_coord = list(map(add, bbox_crop_coord, [crop_xmin, crop_ymin, crop_xmin, crop_ymin]))
            pred_boxes_list.append(bbox_global_coord)

            score = round(score.item(), 4)
            scores_list.append(score)

            pred_classes_list.append(pred_class)

            error_type = num_to_category_dict[pred_class]
            bbox = [image_file_name, error_type]
            bbox.extend(bbox_global_coord)
            bbox.append(score)
            bbox_list.append(bbox)
            """
    return bbox_list#, inst_dict

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--read_img', type=str, default='pil', choices=['pil','path'], help='the way to read img')
    parser.add_argument(
        '--exp_id', type=str, default='test', help='experiment name')
    parser.add_argument(
        '--eval_bs', type=int, default=16, help='batch size when executing eval')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    """
    if args.read_img=='pil':
        #imgs = np.array(Image.open(args.img))#'./cc_1.png')
        imgs = [np.array(Image.open(args.img)), np.array(Image.open(args.img))]
    else:
        #imgs = args.img
        imgs = [args.img, args.img]
    """
    test_data_dir = '/work/u5216579/ctr/data/PCB_v3'
    image_wo_border_dir = os.path.join(test_data_dir, 'images_wo_border')
    label_wo_border_dir = os.path.join(test_data_dir, 'labels_wo_border')
    
    inference_result_image_dir = os.path.join(test_data_dir, 'inference_result', args.exp_id, 'images')
    inference_result_label_dir = os.path.join(test_data_dir, 'inference_result', args.exp_id, 'labels')
    if not os.path.exists(inference_result_image_dir):
        os.makedirs(inference_result_image_dir)
    if not os.path.exists(inference_result_label_dir):
        os.makedirs(inference_result_label_dir)
    img_list = os.listdir(image_wo_border_dir)

    global_start_time = time.time()
    for idx, image_file_name in enumerate(img_list):
        start_time = time.time()
        #bboxes_list, labels_list = list(), list()
        image_file_path = os.path.join(image_wo_border_dir, image_file_name) # big img path
        image_org = cv2.imread(image_file_path)#(args.img)
        crop_rect_list, crop_image_list = crop_sliding_window(image_org)
        
        bbox, label = predictor(model, crop_image_list, args.eval_bs, args.score_thr)#;print('outputs',outputs)
        bbox_list = unify_batch_predictor_output(image_file_name, image_org.shape, crop_rect_list, bbox, label)
        bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        bbox_list = non_max_suppression_slow(bbox_list, 0.5)
        
        csv_to_json(bbox_list, image_wo_border_dir, inference_result_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)

        # Save original image with inference result
        image_bbox = image_org.copy()
        for bbox in bbox_list:
            file_name, error_type, xmin, ymin, xmax, ymax, score = bbox
            color = category_to_color_dict.get(error_type, default_color)
            cv2.rectangle(image_bbox, (xmin, ymin), (xmax, ymax), color, 6)
            cv2.putText(image_bbox, str(score), (xmin, ymin-10), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2, cv2.LINE_8)
        image_bbox_file_path = os.path.join(inference_result_image_dir, image_file_name)
        cv2.imwrite(image_bbox_file_path, image_bbox, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        logger.debug("{:>4d}, crop image number = {:>3d}, time = {:4.3f} s".format(idx, len(crop_rect_list), round(time.time()-start_time, 3)))
    logger.debug("Total time = {} s".format(round(time.time()-global_start_time)))

    #print('bbox_list',bbox_list)
    """
        step = args.eval_bs
        img_bs = [crop_image_list[i:i+step] for i in range(0, len(crop_image_list), step)]

        for imgs in img_bs:
            result = inference_detector(model, imgs)
            #result = inference_detector(model, im)
            #print('@@result: ',result)
            bboxes, labels = parse_out_and_filter_score(result, args.score_thr)
            bboxes_list.extend(bboxes)
            labels_list.extend(labels)
        print('bboxes_list, labels_list', labels_list)
    """
if __name__ == '__main__':
    main()
