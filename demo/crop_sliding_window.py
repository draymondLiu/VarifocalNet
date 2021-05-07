import numpy as np
from collections import OrderedDict
import os, cv2, json
crop_h = 512
crop_w = 512
margin = 100
def csv_to_json(label_array, image_dir, label_json_dir, coord_type="xmin_ymin_w_h", store_score=False):
    """
    Convert csv format to labelme json format.
    Args:
        label_array (list[list] or np.ndarray): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...] format.
                                                (score is optional and is controlled by store_score)
        image_dir (str): directory of image file (file_name = xxx.jpg).
        label_json_dir (str): directory in which json file is stored
        coord_type (str): "xmin_ymin_w_h" or "xmin_ymin_xmax_ymax"
        store_score (boolean): determine if you want to store score in json file or not
    """
    json_dict = OrderedDict()

    if len(label_array) > 0:
        for i, label in enumerate(label_array):
            file_name  = label[0]
            error_type = label[1]

            if coord_type == "xmin_ymin_w_h":
                xmin, ymin, w, h = [int(i) for i in label[2:6]]
                xmax = xmin + w
                ymax = ymin + h
            elif coord_type == "xmin_ymin_xmax_ymax":
                xmin, ymin, xmax, ymax = [int(i) for i in label[2:6]]
            else:
                assert False, 'coord_type should be either xmin_ymin_w_h or xmin_ymin_xmax_ymax'

            if i==0:
                json_dict["version"] = "4.5.6"
                json_dict["flags"] = dict()
                json_dict["shapes"] = list()
                json_dict["imagePath"] = file_name
                json_dict["imageData"] = None

                image_file_path = os.path.join(image_dir, file_name)

                if os.path.isfile(image_file_path):
                    image = cv2.imread(image_file_path)
                    json_dict["imageHeight"] = image.shape[0]
                    json_dict["imageWidth"] = image.shape[1]
                else:
                    logger.warning("{} does not exist".format(image_file_path))
                    return

            shapes = OrderedDict()
            shapes["label"] = error_type
            shapes["points"] = [[xmin, ymin], [xmax, ymax]]
            shapes["group_id"] = None
            shapes["shape_type"] = "rectangle"
            shapes["flags"] = dict()
            if store_score and len(label) >= 7:
                score = float(label[6])
                shapes["score"] = score
            json_dict["shapes"].append(shapes)

        json_file_name = os.path.splitext(file_name)[0] + '.json'
        json_file_path = os.path.join(label_json_dir, json_file_name)
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))

def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    bbox_array = np.array(boxes)
    error_type = bbox_array[:,1].astype(str)
    x1 = bbox_array[:,2].astype(int)
    y1 = bbox_array[:,3].astype(int)
    x2 = bbox_array[:,4].astype(int)
    y2 = bbox_array[:,5].astype(int)
    score = bbox_array[:,6].astype(float)
    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort the bounding boxes by the score (high -> low)
    idxs = np.argsort(score)[::-1]

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the first index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        i = idxs[0]
        pick.append(i)
        suppress = [0]
        # loop over all indexes in the indexes list
        for pos in range(1, len(idxs)):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            # overlap = float(w * h) / (area[i] + area[j] - float(w * h))
            overlap_area = float(w * h)
            overlap = max(overlap_area/area[i], overlap_area/area[j])
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                # x1[i] = min(x1[i], x1[j])
                # y1[i] = min(y1[i], y1[j])
                # x2[i] = max(x2[i], x2[j])
                # y2[i] = max(y2[i], y2[j])
                # bbox_array[i][2:6] = x1[i], y1[i], x2[i], y2[i]
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return [ bbox[0:2] + [int(i) for i in bbox[2:6]] + [float(bbox[6])] for bbox in bbox_array[pick].tolist() ]

def crop_sliding_window(image, crop_h=crop_h, crop_w=crop_w, margin=margin):
    crop_rect_list = list()
    crop_image_list = list()

    image_h, image_w, _ = image.shape

    count_h = image_h // (crop_h - margin)
    count_w = image_w // (crop_w - margin)

    for h in range(0, count_h):
        for w in range(0, count_w):
            crop_ymin = (crop_h - margin) * h
            crop_xmin = (crop_w - margin) * w

            crop_ymax = crop_ymin + crop_h
            crop_xmax = crop_xmin + crop_w

            if crop_ymax > image_h:
                crop_ymax = image_h
                crop_ymin = image_h - crop_h
            if crop_xmax > image_w:
                crop_xmax = image_w
                crop_xmin = image_w - crop_w

            # Store rect and image
            crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
            crop_rect_list.append(crop_rect)

            crop_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            crop_image_list.append(crop_image)
    return crop_rect_list, crop_image_list
