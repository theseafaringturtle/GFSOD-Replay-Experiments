import json
import logging
import os
import math
import argparse
from typing import Dict

import numpy as np
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[10], help='')
    args = parser.parse_args()
    log = logging.getLogger(__name__)

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')

    for shot in args.shot_list:
        # Collect seeds of runs with same k shots
        file_paths = dict()
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue
            # Xseed_Kshot format
            kshot, seed = fname.split('_')
            if kshot != '{}shot'.format(shot):
                continue
            # Seeds may not be contiguous
            seed_int = int(seed.replace("seed", ""))
            file_paths[seed_int] = os.path.join(_dir, 'metrics.json')

        # dict_results maps seed to metrics
        dict_results: Dict[int, dict] = dict()
        for seed, fpath in file_paths.items():
            info_lines = open(fpath).readlines()
            has_final_metrics = False
            for line in info_lines:
                info = json.loads(line)
                # Check that line contains metrics and keep them
                # TODO other info lines could be used to graph loss
                if 'bbox/AP' in info:
                    has_final_metrics = True
                    info.pop('iteration', None)  # not needed
                    # Remove bbox/ part as superfluous
                    keys = list(info.keys())
                    for key in keys:
                        info[key.replace("bbox/", "")] = info[key]
                        info.pop(key)
                    dict_results[seed] = info
            if not has_final_metrics:
                log.error(f"File {fpath} does not contain final metrics, likely didn't finish or errored out")
            # results.append([fid] + [float(x) for x in res_info.split(':')[-1].split(',')])

        if len(dict_results.items()) == 0:
            raise Exception(f"No results could be parsed from {args.res_dir}")

        # Pick metric header keys from any item in dict_results, since all of them should use the same ones. Will determine the order for the array.
        header_keys = list(next(iter(dict_results.values())).keys())

        # Keep using old 2d array with one seed metric per row, collated with numpy.
        # However, there are a few changes apart from using metrics.json instead of log.txt
        # - Seed was used as equivalent of array index since it always went 0-9, that is no longer the case
        # - Unfinished experiments will be detected
        # - It's now possible to retrieve the training loss for graphing
        results = []
        for seed in dict_results.keys():
            metrics = dict_results[seed]
            metrics_array = [seed]
            for key in header_keys:
                metrics_array.append(metrics[key])
            results.append(metrics_array)

        results.sort(key=lambda x: x[0])

        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
        results.append(['μ'] + avg[1:])
        results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header_keys,
            numalign="left",
        )

        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()
