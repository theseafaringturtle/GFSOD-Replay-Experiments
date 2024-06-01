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

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Scale down base LR linearly. Can't hold good performance below 4 GPUs without changing it, and baseline used 8 GPUs.
    num_devices = args.num_gpus if args.num_gpus > 0 else 1
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / (4.0 / num_devices)
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
