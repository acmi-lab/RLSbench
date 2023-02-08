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


NUM_RUNS = 32
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
NUM_GPUS = len(GPU_IDS)
counter = 0

DATASETS = [
    "camelyon",
    "fmow",
    "domainnet",
    "cifar10",
    "cifar100",
    "entity13",
    "entity30",
    "living17",
    "nonliving26",
    "officehome",
    "visda",
]
TARGET_SETS = {
    "cifar10": ["0", "1", "10", "71", "95"],
    "cifar100": ["0", "4", "12", "59", "82"],
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

SEEDS = ["42", "1234"]
ALPHA = ["0.5", "1.0", "3.0", "10.0", "100.0"]
ALGORITHMS = ["TENT"]

SOURCE_FILE = {
    "cifar10": "logs_consistent_erm/cifar10_seed\:%s/ERM-aug-imagenet_pretrained\:imagenet/",
    "cifar100": "logs_consistent_erm/cifar100_seed\:%s/ERM-aug-imagenet_pretrained\:imagenet/",
    "camelyon": "logs_consistent_erm/camelyon_seed\:%s/ERM-aug-rand_pretrained\:rand/",
    "entity13": "logs_consistent_erm/entity13_seed\:%s/ERM-aug-rand_pretrained\:rand/",
    "entity30": "logs_consistent_erm/entity30_seed\:%s/ERM-aug-rand_pretrained\:rand/",
    "living17": "logs_consistent_erm/living17_seed\:%s/ERM-aug-rand_pretrained\:rand/",
    "nonliving26": "logs_consistent_erm/nonliving26_seed\:%s/ERM-aug-rand_pretrained\:rand/",
    "fmow": "logs_consistent_erm/fmow_seed\:%s/ERM-aug-imagenet_pretrained\:imagenet/",
    "domainnet": "logs_consistent_erm/domainnet_seed\:%s/ERM-aug-imagenet_pretrained\:imagenet/",
    "officehome": "logs_consistent_erm/officehome_seed\:%s/ERM-aug-imagenet_pretrained\:imagenet/",
    "visda": "logs_consistent_erm/visda_seed\:%s/ERM-aug-imagenet_pretrained\:imagenet/",
}

gpu_queue = list()
procs = list()
gpu_id = 0
gpu_use = list()

for i in range(NUM_RUNS):
    gpu_queue.append(i % NUM_GPUS)

for algorithm in ALGORITHMS:
    for dataset in DATASETS:
        for seed in SEEDS:
            for alpha in ALPHA:
                for target_set in TARGET_SETS[dataset]:
                    # gpu_id = GPU_IDS[counter % NUM_GPUS]
                    gpu_id = gpu_queue.pop(0)

                    source_model_path = SOURCE_FILE[dataset] % (seed)

                    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python run_expt.py --remote False \
                    --dataset {dataset} --root_dir /home/ubuntu/data --seed {seed} \
                    --transform image_none --algorithm  {algorithm} --test_time_adapt --use_source_model \
                    --source_model_path={source_model_path} --dirichlet_alpha {alpha} \
                    --target_split {target_set} --use_target True  --simulate_label_shift True"

                    print(cmd)

                    procs.append(Popen(cmd, shell=True))
                    gpu_use.append(gpu_id)

                    time.sleep(3)

                    counter += 1

                    if len(procs) == NUM_RUNS:
                        wait = True

                        while wait:
                            done, num = check_for_done(procs)

                            if done:
                                procs.pop(num)
                                wait = False
                                gpu_queue.append(gpu_use.pop(num))
                            else:
                                time.sleep(3)

                        print("\n \n \n \n --------------------------- \n \n \n \n")
                        print(f"{date.today()} - {counter} runs completed")
                        sys.stdout.flush()
                        print("\n \n \n \n --------------------------- \n \n \n \n")

for p in procs:
    p.wait()
procs = []
