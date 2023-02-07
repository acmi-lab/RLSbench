#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_BREEDs.sh <imagenet_dir>"
    exit 1
fi

## Download Imagenet from here
echo "Download Imagenet by registering and following instrutions from http://image-net.org/download-images."

## Download Imagenet hierarchy 
git clone https://github.com/MadryLab/BREEDS-Benchmarks.git
mkdir -p $1/imagenet_class_hierarchy
mv BREEDS-Benchmarks/imagenet_class_hierarchy/modified/*  $1/imagenet_hierarchy/
rm -rf BREEDS-Benchmarks

