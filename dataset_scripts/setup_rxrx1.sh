#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_rxrx1.sh <path_to_dataset>"
    exit 1
fi


python <(echo "from wilds import get_dataset; get_dataset(dataset='rxrx1', download=True, root_dir='$1')")