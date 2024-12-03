import logging
import os
import time

from packaging.version import Version

import torch
from detectron2.engine import launch

from samplers.AblationSampler import AblationSampler
from samplers.ProtoSampler import ProtoSampler
from defrcn.config import get_cfg, set_global_cfg
from defrcn.engine import default_setup, default_argument_parser

import samplers

logger = logging.getLogger(__name__)

# Determinism
if Version(torch.__version__) >= Version("1.11"):
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    logger.warning("Pytorch < 1.11 detected: results will not be deterministic")


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    # Due to bug in yacs' merge_from_list (does not respect set_new_allowed when input is list)
    if args.opts:
        cfg.merge_from_list(args.opts)
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # This is specifically for base classes
    assert "base" in cfg.DATASETS.TRAIN[0]
    # Args
    SamplerClass = getattr(samplers, args.sampler)
    prev_seed = args.prev_seed
    new_seed = args.new_seed
    samples_needed = args.sample_out_size

    sampler: samplers.BaseSampler = SamplerClass(cfg)
    logging.info(f"Using {SamplerClass.__name__}")

    start_time = time.perf_counter()
    sampler.gather_sample_pool(args.sample_pool_size)
    logger.info(f"Enough samples ({args.sample_pool_size}) have been gathered for all classes")
    logger.info(f"Sample gathering time: {time.perf_counter() - start_time} s")

    start_time = time.perf_counter()
    sampler.process_post()
    logger.info(f"Sample pool processing time: {time.perf_counter() - start_time}")

    for s in samples_needed:
        start_time = time.perf_counter()
        filenames_per_class = sampler.select_samples(s)
        logger.info(f"Sample selection time: {time.perf_counter() - start_time}")
        sampler.save(filenames_per_class, s, prev_seed, new_seed)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--sampler', dest='sampler', type=str,
                        help="Type of sampler for base samples. Use BaseAblationSampler for random sampling")
    parser.add_argument('--sample_pool_size', dest='sample_pool_size', default=100, type=int,
                        help="Number of samples to draw prototypes from")
    parser.add_argument('--sample_out_size', dest='sample_out_size', default=[5],
                        type=lambda s: [int(token.strip()) for token in s.split(' ')],
                        help="Number of shots needed as output")
    parser.add_argument('--prev_dataseed', dest='prev_seed', default=0, type=int,
                        help="Int ID of one of your existing splits. This is required for keeping novel classes intact")
    parser.add_argument('--new_dataseed', dest='new_seed', type=int,
                        help="Int ID of output split, e.g. 30. This will add a new split directory or replace an existing one")

    args = parser.parse_args()
    if any(filter(lambda s: s > args.sample_pool_size, args.sample_out_size)):
        raise ValueError("Your sample_pool_size must be greater than every sample_out_size")

    logger.info(f"Shots: {args.sample_out_size}")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
