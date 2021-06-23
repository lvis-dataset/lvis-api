import logging
from lvis import LVIS, LVISResults, LVISEval

# result and val files for 100 randomly sampled images.
ANNOTATION_PATH = "./data/lvis_val_100.json"
RESULT_PATH = "./data/lvis_results_100.json"

ANN_TYPE = 'bbox'

lvis_gt = LVIS(ANNOTATION_PATH)
lvis_dt = LVISResults(lvis_gt,
                      RESULT_PATH,
                      max_dets_per_cat=2,
                      max_dets_per_im=-1)
lvis_eval = LVISEval(lvis_gt, lvis_dt, ANN_TYPE)
lvis_eval.params.max_dets = -1
lvis_eval.params.max_dets_per_cat = 2
lvis_eval.run()
lvis_eval.print_results()
