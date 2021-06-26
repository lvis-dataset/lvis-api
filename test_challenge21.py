from lvis import LVISEval

if __name__ == "__main__":
    # result and val files for 100 randomly sampled images.
    ANNOTATION_PATH = "./data/lvis_val_100.json"
    RESULT_PATH = "./data/lvis_results_100.json"

    lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH, mode="challenge2021")
    lvis_eval.run()
    lvis_eval.print_results()
