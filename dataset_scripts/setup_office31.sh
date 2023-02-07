#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_office31.sh <path_to_dataset>"
    exit 1
fi

curr_dir=$(pwd)
cd $1 

wget https://wjdcloud.blob.core.windows.net/dataset/OFFICE31.zip
unzip OFFICE31.zip
rm OFFICE31.zip

mv OFFICE31 office31

cd $curr_dir
