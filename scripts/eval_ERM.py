import json
import os
import sys
import time
from collections import Counter
from datetime import date
from subprocess import Popen


NUM_RUNS = 12
GPU_IDS = list(range(4))
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
    # 'office31',
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

SEEDS = ["42"]
ALPHA = ["0.5", "1.0", "3.0", "10.0", "100.0"]
ALGORITHMS = ["ERM-aug"]
# ALGORITHMS= ["ERM", "ERM-aug"]

SOURCE_FILE = {
    "cifar10": "logs_consistent/cifar10_seed\:%s/%s-imagenet_pretrained\:imagenet/",
    "cifar100": "logs_consistent/cifar100_seed\:%s/%s-imagenet_pretrained\:imagenet/",
    "camelyon": "logs_consistent/camelyon_seed\:%s/%s-rand_pretrained\:rand/",
    "entity13": "logs_consistent/entity13_seed\:%s/%s-rand_pretrained\:rand/",
    "entity30": "logs_consistent/entity30_seed\:%s/%s-rand_pretrained\:rand/",
    "living17": "logs_consistent/living17_seed\:%s/%s-rand_pretrained\:rand/",
    "nonliving26": "logs_consistent/nonliving26_seed\:%s/%s-rand_pretrained\:rand/",
    "fmow": "logs_consistent/fmow_seed\:%s/%s-imagenet_pretrained\:imagenet/",
    "domainnet": "logs_consistent/domainnet_seed\:%s/%s-imagenet_pretrained\:imagenet/",
    "officehome": "logs_consistent/officehome_seed\:%s/%s-imagenet_pretrained\:imagenet/",
    "visda": "logs_consistent/visda_seed\:%s/%s-imagenet_pretrained\:imagenet/",
}

procs = []

for dataset in DATASETS:
    for seed in SEEDS:
        for alpha in ALPHA:
            for target_set in TARGET_SETS[dataset]:
                for algorithm in ALGORITHMS:
                    gpu_id = GPU_IDS[counter % NUM_GPUS]

                    source_models = SOURCE_FILE[dataset] % (seed, algorithm)

                    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python run_expt.py --remote False \
                    --dataset {dataset} --root_dir /home/ubuntu/data --seed {seed} \
                    --transform image_none --algorithm  {algorithm} --eval_only --use_source_model \
                    --source_model_path={source_models} --dirichlet_alpha {alpha} \
                    --target_split {target_set} --use_target True  --simulate_label_shift True"

                    print(cmd)
                    procs.append(Popen(cmd, shell=True))

                    time.sleep(3)

                    counter += 1

                    if counter % NUM_RUNS == 0:
                        for p in procs:
                            p.wait()
                        procs = []
                        time.sleep(3)

                        print("\n \n \n \n --------------------------- \n \n \n \n")
                        print(f"{date.today()} - {counter} runs completed")
                        sys.stdout.flush()
                        print("\n \n \n \n --------------------------- \n \n \n \n")


for p in procs:
    p.wait()
procs = []
