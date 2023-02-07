#!/bin/bash
counter=0
NUM_RUNS=20

for i in `find . -name "*.png" -type f`; do
    convert $i "${i%.png}".jpg && rm $i &
	
	counter=$((counter+1))
	
	if ! ((counter % NUM_RUNS)); then
		wait
	fi

done

# folder=/tmp/
# find $folder -name "*.png" -exec bash -c 'convert "$1" "${1%.png}".jpg && rm $1' - '{}' +
