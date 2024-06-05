import logging
import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer

from CFATrainer import CFATrainer
from DeFRCNTrainer import DeFRCNTrainer
from MEGA2Trainer import MEGA2Trainer
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

# Original experiment settings
REF_NUM_GPUS = 4
REF_BATCH_SIZE = 16

def setup(args):
    logger = logging.getLogger(__name__)
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Scale down base LR linearly. Can't hold good performance below 4 GPUs without changing it, and baseline used 8 GPUs.
    num_devices = args.num_gpus if args.num_gpus > 0 else 1
    if num_devices != REF_NUM_GPUS or num_devices != REF_NUM_GPUS * 2:
        lr_factor = 1 / (float(REF_NUM_GPUS) / num_devices)
        logger.warning(f"LR scaled by {lr_factor}")
        cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * lr_factor
    # Scale iterations with batch size, so we don't need to change all configs.
    # Batch size will still have to be adjusted by the user in configs/Cbase-RCNN.yaml
    if cfg.SOLVER.IMS_PER_BATCH != REF_BATCH_SIZE:
        batch_factor = 1 / (cfg.SOLVER.IMS_PER_BATCH / float(REF_BATCH_SIZE))
        logger.warning(f"Iterations scaled by {batch_factor}")
        cfg.SOLVER.STEPS = [int(step * batch_factor) for step in cfg.SOLVER.STEPS]
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * batch_factor)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if cfg.TRAINER == "CFATrainer":
        TrainerClass = CFATrainer
    elif cfg.TRAINER == "MEGA2Trainer":
        TrainerClass = MEGA2Trainer
    elif cfg.TRAINER == "DeFRCNTrainer":
        TrainerClass = DeFRCNTrainer
    else:
        raise Exception(f"Unknown trainer: {cfg.TRAINER}")

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
