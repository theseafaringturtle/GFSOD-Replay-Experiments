import os
import json
import re

from detectron2.data import MetadataCatalog

# For the ablation study, restore 5-shot annotations that were cut from COCO in TFA and later works
from detectron2.engine import launch

from defrcn.config import set_global_cfg, get_cfg
from defrcn.engine import default_setup, default_argument_parser
from samplers import AblationSampler

base_classes = MetadataCatalog.get('coco14_trainval_base').get('base_classes', None)
NUM_SHOTS = 10


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
    dummy_sampler = AblationSampler(cfg)

    files_per_class = {cls_id: [] for cls_id in range(len(base_classes))}

    start_seed, end_seed = 0, 9
    for seed in range(start_seed, end_seed + 1):
        split_file_names = [f for f in os.listdir(f'datasets/cocosplit/seed{seed}')]
        for file_name in split_file_names:
            shot, classname = [g for g in re.search('full_box_([0-9]+shot)_(.*)_trainval\.json', file_name).groups()]
            if shot != f'{NUM_SHOTS}shot' or classname not in base_classes:
                continue
            # Read names of jpeg files in the JSON list
            f = open(f'datasets/cocosplit/seed{seed}/{file_name}')
            j = json.load(f)
            f.close()
            image_file_names = [data['file_name'] for data in j['images']]
            image_file_names = [f'datasets/cocosplit/seed{seed}/{file_name}' for file_name in image_file_names]
            class_id = base_classes.index(classname)
            files_per_class[class_id] = image_file_names
            print(f"Collecting {classname}")
        cfg.NEW_DATASEED = seed + 10
        dummy_sampler.save(files_per_class, NUM_SHOTS, prev_seed=seed, new_seed=seed + 10)
        print("Saving finished!")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--new_dataseed', dest='new_seed', type=int,
                        help="Int ID of output split, e.g. 30. This will add a new split directory or replace an existing one")
    args = parser.parse_args()

    launch(
        main,
        num_gpus_per_machine=0,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
