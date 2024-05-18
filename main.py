import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer

from CFATrainer import CFATrainer
from DeFRCNTrainer import DeFRCNTrainer
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
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / (4.0 / args.num_gpus)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if cfg.TRAINER == "CFATrainer":
        TrainerClass = CFATrainer
        # Use joint fine-tuning as in TFA, since model has to process memory samples
        cfg.defrost()
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
        # Replace training set with base-k + novel-k instead of just novel-k
        cfg.DATASETS.TRAIN = (cfg.DATASETS.TRAIN[0].replace("_novel", "_all"), )
        cfg.TEST.PCB_ENABLE = False
        cfg.freeze()
    else:
        TrainerClass = DeFRCNTrainer

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
