import json
import os
import sys
import time
from collections import Counter
from datetime import date
from subprocess import Popen


def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False


NUM_RUNS = 4
GPU_IDS = [0, 1, 2, 3]
NUM_GPUS = len(GPU_IDS)
counter = 0

DATASETS = [
    "domainnet",
    "camelyon",
    # # 'iwildcam',
    "fmow",
    "cifar10",
    "cifar100",
    "entity13",
    "entity30",
    "living17",
    "nonliving26",
    "office31",
    "officehome",
    "visda",
]
TARGET_SETS = {
    "cifar10": ["0", "1", "10", "23", "57", "71", "95"],
    "cifar100": ["0", "4", "12", "43", "59", "82"],
    "fmow": ["0", "1", "2"],
    "iwildcams": ["0", "1", "2"],
    "camelyon": ["0", "1", "2"],
    "domainnet": ["0", "1", "2", "3"],
    "entity13": ["0", "1", "2", "3"],
    "entity30": ["0", "1", "2", "3"],
    "living17": ["0", "1", "2", "3"],
    "nonliving26": ["0", "1", "2", "3"],
    "officehome": ["0", "1", "2", "3"],
    "office31": ["0", "1", "2"],
    "visda": ["0", "1", "2"],
}

SEEDS = ["1234", "42"]
ALPHA = ["0.5", "1.0", "3.0", "10.0", "100.0"]
ALGORITHMS = [
    "DANN",
    "IW-DANN",
    "IW-CDANN",
    "FixMatch",
    "CDANN",
    "SENTRY",
    "IS-DANN",
    "IS-CDANN",
    "IS-FixMatch",
]

procs = list()

for dataset in DATASETS:
    for seed in SEEDS:
        for alpha in ALPHA:
            for algorithm in ALGORITHMS:
                for target_set in TARGET_SETS[dataset]:
                    gpu_id = GPU_IDS[counter % NUM_GPUS]

                    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python run_expt.py --remote False \
                    --dataset {dataset} --root_dir /home/sgarg2/data --seed {seed} \
                    --transform image_none --algorithm  {algorithm}  --dirichlet_alpha {alpha} \
                    --target_split {target_set} --use_target True  --simulate_label_shift True"

                    print(cmd)
                    procs.append(Popen(cmd, shell=True))

                    time.sleep(3)

                    counter += 1

                    if len(procs) == NUM_RUNS:
                        wait = True

                        while wait:
                            done, num = check_for_done(procs)

                            if done:
                                procs.pop(num)
                                wait = False
                            else:
                                time.sleep(3)

                        print("\n \n \n \n --------------------------- \n \n \n \n")
                        print(f"{date.today()} - {counter} runs completed")
                        sys.stdout.flush()
                        print("\n \n \n \n --------------------------- \n \n \n \n")


for p in procs:
    p.wait()
procs = []
