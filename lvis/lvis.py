"""
API for accessing LVIS Dataset: https://lvisdataset.org.

LVIS API is a Python API that assists in loading, parsing and visualizing
the annotations in LVIS. In addition to this API, please download
images and annotations from the LVIS website.
"""

import json
import os
import logging
from collections import defaultdict
from urllib.request import urlretrieve
import time

import pycocotools.mask as mask_utils

from lvis.boundary_utils import augment_annotations_with_boundary_multi_core


class LVIS:
    def __init__(self, annotation_path, precompute_boundary=False, dilation_ratio=0.02, max_cpu_num=80):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): location of annotation file
            precompute_boundary (bool): whether to precompute mask boundary before evaluation
            dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
            max_cpu_num (int): max number of cpu cores to compute mask boundary before evaluation
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")

        self.dataset = self._load_json(annotation_path)

        self.precompute_boundary = precompute_boundary
        self.dilation_ratio = dilation_ratio
        self.max_cpu_num = max_cpu_num

        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        self._create_index()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _create_index(self):
        self.logger.info("Creating index.")

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img
        
        if self.precompute_boundary:
            # add `boundary` to annotation
            self.logger.info('Adding `boundary` to annotation.')
            tic = time.time()
            self.dataset["annotations"] = augment_annotations_with_boundary_multi_core(self.dataset["annotations"],
                                                                                       self.imgs,
                                                                                       dilation_ratio=self.dilation_ratio,
                                                                                       max_cpu_num=self.max_cpu_num)

            self.logger.info('`boundary` added! (t={:0.2f}s)'.format(time.time()- tic))

        for ann in self.dataset["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])

        self.logger.info("Index created.")

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None):
        """Get ann ids that satisfy given filter conditions.

        Args:
            img_ids (int array): get anns for given imgs
            cat_ids (int array): get anns for given cats
            area_rng (float array): get anns for a given area range. e.g [0, inf]

        Returns:
            ids (int array): integer array of ann ids
        """
        anns = []
        if img_ids is not None:
            for img_id in img_ids:
                anns.extend(self.img_ann_map[img_id])
        else:
            anns = self.dataset["annotations"]

        # return early if no more filtering required
        if cat_ids is None and area_rng is None:
            return [_ann["id"] for _ann in anns]

        cat_ids = set(cat_ids)

        if area_rng is None:
            area_rng = [0, float("inf")]

        ann_ids = [
            _ann["id"]
            for _ann in anns
            if _ann["category_id"] in cat_ids
            and _ann["area"] > area_rng[0]
            and _ann["area"] < area_rng[1]
        ]
        return ann_ids

    def get_cat_ids(self):
        """Get all category ids.

        Returns:
            ids (int array): integer array of category ids
        """
        return list(self.cats.keys())

    def get_img_ids(self):
        """Get all img ids.

        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def _load_helper(self, _dict, ids):
        if ids is None:
            return list(_dict.values())
        else:
            return [_dict[id] for id in ids]

    def load_anns(self, ids=None):
        """Load anns with the specified ids. If ids=None load all anns.

        Args:
            ids (int array): integer array of annotation ids

        Returns:
            anns (dict array) : loaded annotation objects
        """
        return self._load_helper(self.anns, ids)

    def load_cats(self, ids):
        """Load categories with the specified ids. If ids=None load all
        categories.

        Args:
            ids (int array): integer array of category ids

        Returns:
            cats (dict array) : loaded category dicts
        """
        return self._load_helper(self.cats, ids)

    def load_imgs(self, ids):
        """Load categories with the specified ids. If ids=None load all images.

        Args:
            ids (int array): integer array of image ids

        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.imgs, ids)

    def download(self, save_dir, img_ids=None):
        """Download images from mscoco.org server.
        Args:
            save_dir (str): dir to save downloaded images
            img_ids (int array): img ids of images to download
        """
        imgs = self.load_imgs(img_ids)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for img in imgs:
            file_name = os.path.join(save_dir, img["coco_url"].split("/")[-1])
            if not os.path.exists(file_name):
                urlretrieve(img["coco_url"], file_name)
