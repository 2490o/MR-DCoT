"""Guarded MR-DCoT training entry for the DATAC baseline.

The model starts from the same initialization path as train_datac.py. During
training it evaluates all weather domains every 50000 iterations, and after
training it reloads model_best.pth for a final all-domain markdown report.
"""

import logging
import os
from collections import OrderedDict

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import HookBase, default_argument_parser, default_setup, hooks
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage

import train_datac as base
import modeling.meta_arch_datac_mrdcot  # noqa: F401
from modeling import CustomPascalVOCDetectionEvaluator


logger = logging.getLogger("detectron2")

TARGET_DOMAIN_ORDER = [
    ("night_sunny", "night_sunny_test"),
    ("dusk_rainy", "dusk_rainy_train"),
    ("night_rainy", "night_rainy_train"),
    ("daytime_foggy", "daytime_foggy_train"),
    ("daytime_clear", "daytime_clear_test"),
]

RESULT_MD = "/home/zzh/SE-COT/all_outs/eval_results.md"


def add_guard_config(cfg):
    cfg.TEST.MILESTONE_EVAL_PERIOD = 50000
    cfg.TEST.FINAL_BEST_ALL_DOMAIN = True

    cfg.MODEL.MR_GUARD_ENABLE = True
    cfg.MODEL.MR_GUARD_FORCE_BASELINE = False
    cfg.MODEL.MR_GUARD_START_ITER = 100000
    cfg.MODEL.MR_GUARD_WARMUP_ITERS = 50000
    cfg.MODEL.MR_GUARD_MAX_ALPHA = 0.05
    cfg.MODEL.MR_GUARD_GATE_INIT = -6.0

    cfg.MODEL.MR_GUARD_PROTO_DIM = 256
    cfg.MODEL.MR_GUARD_TEMPERATURE = 0.1
    cfg.MODEL.MR_GUARD_LOCAL_WEIGHT = 0.5
    cfg.MODEL.MR_GUARD_SOURCE_PROTO_WEIGHT = 0.05
    cfg.MODEL.MR_GUARD_MAX_GT_ROIS = 64

    cfg.MODEL.MR_GUARD_VISUAL_NOISE_STD = 0.03
    cfg.MODEL.MR_GUARD_BLUR_KERNEL = 3
    cfg.MODEL.MR_GUARD_VISUAL_STEPS = 2
    cfg.MODEL.MR_GUARD_DROP_PROB = 0.0
    cfg.MODEL.MR_GUARD_STYLE_MEAN_SCALE = 0.5
    cfg.MODEL.MR_GUARD_STYLE_STD_SCALE = 0.5

    cfg.MODEL.MR_GUARD_LOSS_ALIGN = 0.01
    cfg.MODEL.MR_GUARD_LOSS_DIFF = 0.005
    cfg.MODEL.MR_GUARD_LOSS_REG = 0.02
    cfg.MODEL.MR_GUARD_LOSS_OFF_DET = 0.0
    cfg.MODEL.MR_GUARD_OFF_DET_MAX_IMAGES = 2


def setup(args):
    cfg = get_cfg()
    base.add_stn_config(cfg)
    add_guard_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def append_or_update_markdown_row(cfg, results, weight_name, result_path=RESULT_MD):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    exp_name = os.path.basename(os.path.normpath(cfg.OUTPUT_DIR))
    values = {}
    for dataset_name, result in results.items():
        if "bbox" in result and "AP50" in result["bbox"]:
            values[dataset_name] = result["bbox"]["AP50"]

    row = [exp_name, weight_name]
    for _, dataset_name in TARGET_DOMAIN_ORDER:
        ap50 = values.get(dataset_name)
        row.append("-" if ap50 is None else f"{ap50:.2f}")
    row_text = "| " + " | ".join(row) + " |\n"

    header = (
        "| experiment | weight | night_sunny | dusk_rainy | night_rainy | daytime_foggy | daytime_clear |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    key = f"| {exp_name} | {weight_name} |"

    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        with open(result_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = header.splitlines(True)

    if not lines or not lines[0].startswith("| experiment |"):
        lines = header.splitlines(True) + lines

    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith(key):
            lines[idx] = row_text
            replaced = True
            break

    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(row_text)

    with open(result_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info("AP50 markdown row written to %s: %s", result_path, row_text.strip())


def evaluate_target_domains(cfg, model, weight_name):
    results = OrderedDict()
    for _, dataset_name in TARGET_DOMAIN_ORDER:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = CustomPascalVOCDetectionEvaluator(dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i

        if comm.is_main_process():
            logger.info("Evaluation results for %s in csv format:", dataset_name)
            print_csv_format(results_i)
            if "bbox" in results_i and "AP50" in results_i["bbox"]:
                try:
                    storage = get_event_storage()
                    storage.put_scalar(
                        f"{weight_name}_{dataset_name}_AP50",
                        results_i["bbox"]["AP50"],
                        smoothing_hint=False,
                    )
                except Exception:
                    pass

    if comm.is_main_process():
        append_or_update_markdown_row(cfg, results, weight_name)
    return results


class PeriodicAllDomainEvalHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.period = int(cfg.TEST.MILESTONE_EVAL_PERIOD)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self.period <= 0 or next_iter % self.period != 0:
            return

        logger.info("Running all-domain guarded evaluation at iter %d", next_iter)
        evaluate_target_domains(self.cfg, self.trainer.model, f"iter_{next_iter}")


class Trainer(base.Trainer):
    pass


def test_best_checkpoint(cfg, model):
    if not bool(cfg.TEST.FINAL_BEST_ALL_DOMAIN):
        return None

    best_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    if not os.path.exists(best_path):
        logger.warning("Skip final all-domain test: %s does not exist.", best_path)
        return None

    logger.info("Loading best checkpoint for final all-domain test: %s", best_path)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(best_path, resume=False)

    eval_cfg = cfg.clone()
    eval_cfg.defrost()
    eval_cfg.DATASETS.TEST = tuple(dataset_name for _, dataset_name in TARGET_DOMAIN_ORDER)
    eval_cfg.MODEL.WEIGHTS = best_path
    return evaluate_target_domains(eval_cfg, model, "model_best.pth")


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return base.do_test(cfg, model)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    for dataset_name in cfg.DATASETS.TEST:
        if (
            "daytime_clear_test" in dataset_name
            or "dusk_rainy_train" in dataset_name
            or "night_sunny_test" in dataset_name
            or "night_rainy_train" in dataset_name
            or "daytime_foggy_train" in dataset_name
        ):
            trainer.register_hooks(
                [
                    hooks.BestCheckpointer(
                        cfg.TEST.EVAL_SAVE_PERIOD,
                        trainer.checkpointer,
                        f"{dataset_name}_AP50",
                        file_prefix="model_best",
                    )
                ]
            )

    trainer.register_hooks([PeriodicAllDomainEvalHook(cfg)])
    trainer.train()
    return test_best_checkpoint(cfg, trainer.model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
