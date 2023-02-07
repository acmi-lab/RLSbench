#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_Imagenet.sh <imagenet_dir>"
    exit 1
fi

if [ -f "$1/imagenet_class_hierarchy/dataset_class_info.json" ]
then
    echo "OK"
else
   echo "Please download the BREEDs heirarcy first with the following command:"
   echo "./setup_BREEDs.sh ${1}"
fi

## Download Imagenet from here
echo "Download Imagenet by registering and following instrutions from http://image-net.org/download-images."

## Download Imagenetv2
echo "Downloading Imagenetv2..."
mkdir -p $1/imagenetv2

wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz  -C  $1/imagenetv2/  
rm -rf  imagenetv2-matched-frequency.tar.gz
python dataset_setup/Imagenet/ImageNet_v2_reorg.py --dir $1/imagenetv2/imagenetv2-matched-frequency-format-val --info $1/imagenet_class_hierarchy/dataset_class_info.json

wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz
tar -xvf imagenetv2-threshold0.7.tar.gz -C  $1/imagenetv2/
rm -rf imagenetv2-threshold0.7.tar.gz
python dataset_setup/Imagenet/ImageNet_v2_reorg.py --dir $1/imagenetv2/imagenetv2-threshold0.7-format-val --info $1/imagenet_class_hierarchy/dataset_class_info.json


wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz
tar -xvf imagenetv2-top-images.tar.gz -C  $1/imagenetv2/
rm -rf imagenetv2-top-images.tar.gz
python dataset_setup/Imagenet/ImageNet_v2_reorg.py --dir $1/imagenetv2/imagenetv2-top-images-format-val --info $1/imagenet_class_hierarchy/dataset_class_info.json

echo "Imagenetv2 downloaded"

## Download Imagenet C
echo "Downloading Imagenet C..."
mkdir -p $1/imagenet-c

wget https://zenodo.org/record/2235448/files/blur.tar?download=1 
tar -xvf "blur.tar?download=1" -C  $1/imagenet-c/
rm -rf "blur.tar?download=1"

wget https://zenodo.org/record/2235448/files/digital.tar?download=1
tar -xvf "digital.tar?download=1" -C  $1/imagenet-c/
rm -rf "digital.tar?download=1"

wget https://zenodo.org/record/2235448/files/extra.tar?download=1
tar -xvf "extra.tar?download=1" -C  $1/imagenet-c/
rm -rf "extra.tar?download=1"

wget https://zenodo.org/record/2235448/files/noise.tar?download=1
tar -xvf "noise.tar?download=1" -C  $1/imagenet-c/
rm -rf "noise.tar?download=1"

wget https://zenodo.org/record/2235448/files/weather.tar?download=1
tar -xvf "weather.tar?download=1" -C  $1/imagenet-c/
rm -rf "weather.tar?download=1"

echo "Imagenet C downloaded"

## Download Imagenet R

echo "Downloading Imagenet R..."
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvf imagenet-r.tar -C  $1/
rm -rf imagenet-r.tar

echo "Imagenet R downloaded"

## Download Imagenet Sketch

echo "Downloading Imagenet Sketch..."
gdown https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
unzip ImageNet-Sketch.zip -d $1/
mv $1/sketch/ $1/imagenet-sketch/
rm -rf ImageNet-Sketch.zip

echo "Imagenet Sketch downloaded"
