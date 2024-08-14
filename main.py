import logging
import os
import re

from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer

from AGEMTrainer import AGEMTrainer
from CFALTrainer import CFALTrainer
from CFATrainer import CFATrainer
from DeFRCNTrainer import DeFRCNTrainer
from DebugTrainer import DebugTrainer
from EWCTrainer import EWCTrainer
from GPMTrainer import GPMTrainer
from MEGA2Trainer import MEGA2Trainer
from MemoryTrainer import MemoryTrainer
from ERTrainer import ERTrainer
from CAGTrainer import CAGTrainer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import verify_results
from defrcn.engine import default_argument_parser, default_setup

import torch

# Determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if "1.8.2" in torch.__version__:
    torch.use_deterministic_algorithms(True)
else:
    torch.use_deterministic_algorithms(True, warn_only=True)

logger = logging.getLogger(__name__)

# Original experiment settings
REF_NUM_GPUS = 4
REF_BATCH_SIZE = 16


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Scale iterations and LR with batch size, so we don't need to change all configs.
    # Batch size will still have to be adjusted by the user in configs/Base-RCNN.yaml
    if cfg.SOLVER.IMS_PER_BATCH != REF_BATCH_SIZE:
        logger.warning(
            f"Batch size of {cfg.SOLVER.IMS_PER_BATCH} instead of original {REF_BATCH_SIZE}. Adjusting hyperparams")
        batch_factor = 1 / (cfg.SOLVER.IMS_PER_BATCH / float(REF_BATCH_SIZE))
        logger.warning(f"Iterations multiplied by by {batch_factor}")
        cfg.SOLVER.STEPS = [int(step * batch_factor) for step in cfg.SOLVER.STEPS]
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * batch_factor)
        logger.warning(f"Learning Rate multiplied by {1 / batch_factor}")
        cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * 1 / batch_factor
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    cfg.defrost()
    TrainerClass = globals()[cfg.TRAINER]
    # Using DeFRCNTrainer as superclass for all trainers. Not strictly required.
    assert issubclass(TrainerClass, DeFRCNTrainer)

    cfg.freeze()
    if args.eval_only:
        model = TrainerClass.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = TrainerClass.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = TrainerClass(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
