"""
Training script for the independent MR-DCoT experimental branch.

It keeps the original train_cot.py untouched and registers ClipRCNNMRDCOT.
"""

import os
import logging
import time
from collections import OrderedDict, Counter
import copy

import numpy as np
import torch
import torch.utils.data as torchdata

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.engine import default_argument_parser, hooks, HookBase
from detectron2.solver.build import maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm
from detectron2.evaluation import print_csv_format, inference_on_dataset
from detectron2.solver import LRMultiplier
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as detT
import detectron2.data.detection_utils as utils

import torchvision.transforms as T
import torchvision.transforms.functional as tF

from fvcore.common.param_scheduler import ParamScheduler

from data.datasets import builtin  # noqa: F401
from modeling import meta_arch_mrdcot  # noqa: F401
from modeling.config import add_stn_config
from modeling.custom_pascal_evaluation import CustomPascalVOCDetectionEvaluator

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
        self.with_crops = cfg.INPUT.CLIP_WITH_IMG
        self.with_random_clip_crops = cfg.INPUT.CLIP_RANDOM_CROPS
        self.with_jitter = cfg.INPUT.IMAGE_JITTER
        self.cropfn = T.RandomCrop
        self.aug = T.ColorJitter(brightness=0.5, hue=0.3)
        self.crop_size = cfg.INPUT.RANDOM_CROP_SIZE
        self.crop_num = cfg.INPUT.RANDOM_CROP_NUM

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = detT.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if self.with_jitter:
            dataset_dict["jitter_image"] = self.aug(dataset_dict["image"])

        if self.with_crops:
            bbox = dataset_dict["instances"].gt_boxes.tensor
            csx = (bbox[:, 0] + bbox[:, 2]) * 0.5
            csy = (bbox[:, 1] + bbox[:, 3]) * 0.5
            maxwh = torch.maximum(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])
            crops, gt_boxes = [], []
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            for cx, cy, maxdim, box in zip(csx, csy, maxwh, bbox):
                if int(maxdim) < 10:
                    continue
                x0 = torch.maximum(cx - maxdim * 0.5, torch.tensor(0))
                y0 = torch.maximum(cy - maxdim * 0.5, torch.tensor(0))
                imcrop = T.functional.resized_crop(
                    dataset_dict["image"], top=int(y0), left=int(x0),
                    height=int(maxdim), width=int(maxdim), size=224,
                )
                imcrop = T.functional.normalize(imcrop.flip(0) / 255, mean, std)
                crops.append(imcrop.unsqueeze(0))
                gt_boxes.append(box.reshape(1, -1))
            dataset_dict["crops"] = (
                [torch.cat(crops, 0), gt_boxes] if crops else []
            )

        if self.with_random_clip_crops:
            crops, rbboxs = [], []
            for _ in range(self.crop_num):
                p = self.cropfn.get_params(dataset_dict["image"], [self.crop_size, self.crop_size])
                c = tF.crop(dataset_dict["image"], *p)
                if self.crop_size != 224:
                    c = tF.resize(c, size=224)
                crops.append(c)
                rbboxs.append(p)
            dataset_dict["randomcrops"] = torch.stack(crops)

            if self.with_jitter:
                jitter_crops = []
                for p in rbboxs:
                    jc = tF.crop(dataset_dict["jitter_image"], *p)
                    if self.crop_size != 224:
                        jc = tF.resize(jc, size=224)
                    jitter_crops.append(jc)
                dataset_dict["jitter_randomcrops"] = torch.stack(jitter_crops)

        return dataset_dict


class Trainer(DefaultTrainer):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.off_opt_interval = list(
            range(0, cfg.SOLVER.MAX_ITER, cfg.OFFSET_OPT_INTERVAL[0])
        )
        self.off_opt_iters = cfg.OFFSET_OPT_ITERS

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if MetadataCatalog.get(dataset_name).evaluator_type == "pascal_voc":
            return CustomPascalVOCDetectionEvaluator(dataset_name)
        from detectron2.evaluation import COCOEvaluator
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        trainable = {"detector": [], "style": []}
        for name, val in model.named_parameters():
            if not val.requires_grad:
                continue
            if any(k in name for k in ("stylemean", "stylestd", "e_style", "e_content")):
                trainable["style"].append(val)
            else:
                trainable["detector"].append(val)

        optimizer_det = torch.optim.SGD(
            trainable["detector"],
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        optimizer_style = torch.optim.SGD(
            [p for n, p in model.named_parameters() if "stylemean" in n or "stylestd" in n],
            cfg.OFFSET_OPT_LR,
            momentum=0.9,
            weight_decay=0.0005,
        )
        return (
            maybe_add_gradient_clipping(cfg, optimizer_det),
            maybe_add_gradient_clipping(cfg, optimizer_style),
        )

    def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        opt_phase = False
        if (
            self.off_opt_iters > 0
            and len(self.off_opt_interval)
            and self.iter >= self.off_opt_interval[0]
            and self.iter < self.off_opt_interval[0] + self.off_opt_iters
        ):
            if self.iter == self.off_opt_interval[0]:
                self.model.stylemean.data.zero_()
                self.model.stylestd.data.fill_(1.0)
            loss_dict_s = self.model.opt_offsets(data)
            opt_phase = True
            if self.iter + 1 == self.off_opt_interval[0] + self.off_opt_iters:
                self.off_opt_interval.pop(0)
        else:
            loss_dict_s = self.model(data)

        loss = sum(loss_dict_s.values())
        self.optimizer[0].zero_grad()
        self.optimizer[1].zero_grad()
        loss.backward()

        if opt_phase:
            self.optimizer[1].step()
        else:
            self.optimizer[0].step()

        self.optimizer[0].zero_grad()
        self.optimizer[1].zero_grad()
        self._trainer._write_metrics(loss_dict_s, data_time)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            LRScheduler(),
        ]

        if comm.is_main_process():
            ret.append(OverwriteLastCheckpointer(self.checkpointer, cfg.TEST.EVAL_SAVE_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            put_dataset_ap50_scalars(self.cfg, self._last_eval_results)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer[0])

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer_det"] = self.optimizer[0].state_dict()
        ret["optimizer_style"] = self.optimizer[1].state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "optimizer_det" in state_dict:
            self.optimizer[0].load_state_dict(state_dict["optimizer_det"])
        if "optimizer_style" in state_dict:
            self.optimizer[1].load_state_dict(state_dict["optimizer_style"])


class OverwriteLastCheckpointer(HookBase):
    def __init__(self, checkpointer, period):
        self.checkpointer = checkpointer
        self.period = int(period)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self.period > 0 and next_iter % self.period == 0:
            self.checkpointer.save("model_last", iteration=self.trainer.iter)

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self.checkpointer.save("model_last", iteration=self.trainer.iter)


class LRScheduler(HookBase):
    def __init__(self, optimizer=None, scheduler=None):
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_id_det = self._best_param_group_id(self._optimizer[0])
        self._best_id_style = self._best_param_group_id(self._optimizer[1])

    @staticmethod
    def _best_param_group_id(optimizer):
        largest = max(len(g["params"]) for g in optimizer.param_groups)
        for i, g in enumerate(optimizer.param_groups):
            if len(g["params"]) == largest:
                return i
        return 0

    def after_step(self):
        storage = self.trainer.storage
        storage.put_scalar(
            "lr_det",
            self._optimizer[0].param_groups[self._best_id_det]["lr"],
            smoothing_hint=False,
        )
        storage.put_scalar(
            "lr_style",
            self._optimizer[1].param_groups[self._best_id_style]["lr"],
            smoothing_hint=False,
        )
        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = CustomPascalVOCDetectionEvaluator(dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def put_dataset_ap50_scalars(cfg, results):
    if not comm.is_main_process():
        return

    if len(cfg.DATASETS.TEST) == 1 and "bbox" in results:
        results_by_dataset = {cfg.DATASETS.TEST[0]: results}
    else:
        results_by_dataset = results

    storage = get_event_storage()
    for dataset_name, results_i in results_by_dataset.items():
        if "bbox" in results_i and "AP50" in results_i["bbox"]:
            storage.put_scalar(
                f"{dataset_name}_AP50",
                results_i["bbox"]["AP50"],
                smoothing_hint=False,
            )


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if len(cfg.DATASETS.TEST) > 0:
        dataset_name = cfg.DATASETS.TEST[0]
        trainer.register_hooks([
            hooks.BestCheckpointer(
                cfg.TEST.EVAL_SAVE_PERIOD,
                trainer.checkpointer,
                f"{dataset_name}_AP50",
                file_prefix="model_best",
            ),
        ])

    trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
