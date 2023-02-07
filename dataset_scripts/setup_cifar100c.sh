#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_cifar100c.sh <cifar_dir>"
    exit 1
fi

## Download CIFAR10-C
echo "Downloading CIFAR-100 C..."

wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf "CIFAR-100-C.tar?download=1" -C  $1/
rm -rf "CIFAR-100-C.tar?download=1"

echo "CIFAR100-C downloaded"
