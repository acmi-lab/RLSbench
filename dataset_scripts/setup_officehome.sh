#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_officehome.sh <path_to_dataset>"
    exit 1
fi

curr_dir=$(pwd)


mkdir $1/ 
cd $1/

wget https://wjdcloud.blob.core.windows.net/dataset/OfficeHome.zip
unzip OfficeHome.zip
rm OfficeHome.zip

mv OfficeHome officehome

cd $curr_dir