from copy import deepcopy
import logging
from collections import defaultdict
from lvis.lvis import LVIS

import pycocotools.mask as mask_utils


class LVISResults(LVIS):
    def __init__(
        self,
        lvis_gt,
        results,
        max_dets_per_cat=-1,
        max_dets_per_im=300,
        precompute_boundary=False,
        dilation_ratio=0.02,
    ):
        """Constructor for LVIS results.
        Args:
            lvis_gt (LVIS class instance, or str containing path of
            annotation file)
            results (str containing path of result file or a list of dicts)
            max_dets_per_cat (int):  max number of detections per category. The
                official value for the current version of the LVIS API is
                infinite (i.e., -1).  The official value for the 2021 LVIS
                challenge is 10,000.
            max_dets_per_im (int):  max number of detections per image. The
                official value for the current version of the LVIS API is 300.
                The official value for the 2021 LVIS challenge is infinite
                (i.e., -1).
            precompute_boundary (bool): whether to precompute mask boundary before evaluation
            dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        """
        if isinstance(lvis_gt, LVIS):
            self.dataset = deepcopy(lvis_gt.dataset)
            precompute_boundary = lvis_gt.precompute_boundary
        elif isinstance(lvis_gt, str):
            self.dataset = self._load_json(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))

        self.precompute_boundary = precompute_boundary
        self.dilation_ratio = dilation_ratio

        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading and preparing results.")

        if isinstance(results, str):
            result_anns = self._load_json(results)
        else:
            # this path way is provided to avoid saving and loading result
            # during training.
            self.logger.warn("Assuming user provided the results in correct format.")
            result_anns = results

        assert isinstance(result_anns, list), "results is not a list."

        if max_dets_per_im >= 0:
            result_anns = self.limit_dets_per_image(result_anns, max_dets_per_im)
        self.max_dets_per_im = max_dets_per_im
        self.max_dets_per_cat = max_dets_per_cat
        if max_dets_per_cat >= 0:
            result_anns = self.limit_dets_per_cat(result_anns, max_dets_per_cat)

        if "bbox" in result_anns[0]:
            for id, ann in enumerate(result_anns):
                x1, y1, w, h = ann["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann["area"] = w * h
                ann["id"] = id + 1

        elif "segmentation" in result_anns[0]:
            for id, ann in enumerate(result_anns):
                # Only support compressed RLE format as segmentation results
                ann["area"] = mask_utils.area(ann["segmentation"])

                if "bbox" not in ann:
                    ann["bbox"] = mask_utils.toBbox(ann["segmentation"])

                ann["id"] = id + 1

        self.dataset["annotations"] = result_anns
        self._create_index()

        img_ids_in_result = [ann["image_id"] for ann in result_anns]

        assert set(img_ids_in_result) == (
            set(img_ids_in_result) & set(self.get_img_ids())
        ), "Results do not correspond to current LVIS set."

    def limit_dets_per_cat(self, anns, max_dets):
        by_cat = defaultdict(list)
        for ann in anns:
            by_cat[ann["category_id"]].append(ann)
        results = []
        fewer_dets_cats = set()
        for cat, cat_anns in by_cat.items():
            if len(cat_anns) < max_dets:
                fewer_dets_cats.add(cat)
            results.extend(
                sorted(cat_anns, key=lambda x: x["score"], reverse=True)[:max_dets]
            )
        if fewer_dets_cats:
            self.logger.warning(
                f"{len(fewer_dets_cats)} categories had less than {max_dets} "
                f"detections!\n"
                f"Outputting {max_dets} detections for each category will improve AP "
                f"further."
            )
        return results

    def limit_dets_per_image(self, anns, max_dets):
        img_ann = defaultdict(list)
        for ann in anns:
            img_ann[ann["image_id"]].append(ann)

        for img_id, _anns in img_ann.items():
            if len(_anns) <= max_dets:
                continue
            _anns = sorted(_anns, key=lambda ann: ann["score"], reverse=True)
            img_ann[img_id] = _anns[:max_dets]

        return [ann for anns in img_ann.values() for ann in anns]

    def get_top_results(self, img_id, score_thrs):
        ann_ids = self.get_ann_ids(img_ids=[img_id])
        anns = self.load_anns(ann_ids)
        return list(filter(lambda ann: ann["score"] > score_thrs, anns))
