#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_visda.sh <path_to_dataset> <path_to_visda_script>"
    exit 1
fi

curr_dir=$(pwd)
mkdir $1/visda
cd $1/visda 

wget http://csr.bu.edu/ftp/visda17/clf/train.tar
tar xvf train.tar

wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar  

wget http://csr.bu.edu/ftp/visda17/clf/test.tar
tar xvf test.tar

wget https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt 

python $2 --dir=./test/ --map=image_list.txt

rm -rf test/trunk*

rm -rf train.tar validation.tar test.tar

cd $curr_dir
