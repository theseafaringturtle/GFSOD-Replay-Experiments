import os
import random

import numpy as np
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_voc"]

DATASET_BASE_DIR = 'datasets'

VOC_BASE_INSTANCE_CAP = os.environ.get("VOC_BASE_INSTANCE_CAP", None) == "true"

def load_filtered_voc_instances(
    name: str, dirname: str, split: str, classnames: str
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    is_shots = "shot" in name
    if is_shots:
        if not os.path.exists(DATASET_BASE_DIR):
            raise FileNotFoundError(f"./{DATASET_BASE_DIR} should be a symlink or directory containing 'vocsplit'")
        split_dir = os.path.join(DATASET_BASE_DIR, "vocsplit")
        shot = name.split("_")[-2].split("shot")[0]
        seed = int(name.split("_seed")[-1])
        split_dir = os.path.join(split_dir, "seed{}".format(seed))
        fileids = {}
        for cls in classnames:
            with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
            ) as f:
                fileids_ = np.loadtxt(f, dtype=str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                ]
                fileids[cls] = fileids_
    else:
        with PathManager.open(
            os.path.join(dirname, "ImageSets", "Main", split + ".txt")
        ) as f:
            fileids = np.loadtxt(f, dtype=str)

    dicts = []
    instance_counts = [0 for cls_id in range(len(classnames))]
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            offset = 0 # class ID offset, in case we're transposing the IDs of a novel-only set onto 'all'
            for fileid in fileids_:
                year = "2012" if "_" in fileid else "2007"
                dirname = os.path.join(DATASET_BASE_DIR, "VOC{}".format(year))
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname, "JPEGImages", fileid + ".jpg"
                )

                tree = ET.parse(anno_file)

                # If using a memory-based method, don't start from 0 as we'll later use the both novel and base classes separately for the same head
                if "novel_mem" in name:
                    offset = 15
                else:
                    offset = 0

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        if VOC_BASE_INSTANCE_CAP or classnames.index(cls_) >= 15:
                            continue
                    instance_counts[classnames.index(cls_)] += 1
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls_) + offset,
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)
            # Only novel extra instances should be clipped to respect x-shot constraints,
            # unless we're doing things the old TFA/DeFRCN way, that is clipping base ones too
            if (classnames.index(cls) + offset >= 15 or VOC_BASE_INSTANCE_CAP) and instance_counts[classnames.index(cls)] > int(shot):
                d = []
                count = 0
                random.shuffle(dicts_)
                while count < int(shot):
                    anno = dicts_.pop(0)
                    if any(map(lambda x: x['category_id'] != classnames.index(cls) and (x['category_id'] >= 15 or VOC_BASE_INSTANCE_CAP), anno['annotations'])):
                        continue
                    d.append(anno)
                    count += len([d for d in anno['annotations'] if d['category_id'] == classnames.index(cls)])
                dicts_ = d
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append(
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                )
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_voc(
    name, metadata, dirname, split, year, keepclasses, sid
):
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_voc_instances(
            name, dirname, split, thing_classes
        ),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=year,
        split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid],
    )
