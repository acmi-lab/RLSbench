#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_cifar10c.sh <cifar_dir>"
    exit 1
fi


## Download CIFAR10v2
echo "Downloading CIFAR10v2..."
mkdir -p $1/cifar10v2

wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_train.npz
mv cifar102_train.npz $1/cifar10v2/

wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_test.npz
mv cifar102_test.npz $1/cifar10v2/

echo "CIFAR10v2 downloaded"

## Download CIFAR10-C
echo "Downloading CIFAR-10 C..."

wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf "CIFAR-10-C.tar?download=1" -C  $1/
rm -rf "CIFAR-10-C.tar?download=1"

echo "CIFAR10-C downloaded"
