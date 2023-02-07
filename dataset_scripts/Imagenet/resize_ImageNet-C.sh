TYPES=("shot_noise" "impulse_noise" "contrast" "elastic_transform" "pixelate" "jpeg_compression" "speckle_noise" "spatter" "gaussian_blur" "saturate")

for type in "${TYPES[@]}"; do
	echo "${type}"
	command="python ImageNet_resize.py --dir=data/ImageNet/ImageNet-C/${type}/"
	eval $command 
done