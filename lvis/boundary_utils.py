import cv2
import logging
import multiprocessing
import numpy as np

import pycocotools.mask as mask_utils

logger = logging.getLogger(__name__)

MAX_CPU_NUM = 80


def ann_to_rle(ann, imgs):
    """Convert annotation which can be polygons, uncompressed RLE to RLE.
    Args:
        ann (dict) : annotation object

    Returns:
        ann (rle)
    """
    img_data = imgs[ann["image_id"]]
    h, w = img_data["height"], img_data["width"]
    segm = ann["segmentation"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann["segmentation"]
    return rle


def ann_to_mask(ann, imgs):
    """Convert annotation which can be polygons, uncompressed RLE, or RLE
    to binary mask.
    Args:
        ann (dict) : annotation object

    Returns:
        binary mask (numpy 2D array)
    """
    rle = ann_to_rle(ann, imgs)
    return mask_utils.decode(rle)


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


# COCO/LVIS related util functions, to get the boundary for every annotations.
def augment_annotations_with_boundary_single_core(proc_id, annotations, imgs, dilation_ratio=0.02):
    new_annotations = []

    for ann in annotations:
        mask = ann_to_mask(ann, imgs)
        # Find mask boundary.
        boundary = mask_to_boundary(mask, dilation_ratio)
        # Add boundary to annotation in RLE format.
        ann['boundary'] = mask_utils.encode(
            np.array(boundary[:, :, None], order="F", dtype="uint8"))[0]
        new_annotations.append(ann)
    
    return new_annotations


def augment_annotations_with_boundary_multi_core(annotations, imgs, dilation_ratio=0.02):
    cpu_num = min(multiprocessing.cpu_count(), MAX_CPU_NUM)
    annotations_split = np.array_split(annotations, cpu_num)
    logger.info("Number of cores: {}, annotations per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(augment_annotations_with_boundary_single_core,
                                (proc_id, annotation_set, imgs, dilation_ratio))
        processes.append(p)
    
    new_annotations = []
    for p in processes:
        new_annotations.extend(p.get())
    
    workers.close()
    workers.join()
    
    return new_annotations
