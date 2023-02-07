import numpy 
import argparse
import os 

#VISDA classes 
classes = ["aeroplane","bicycle","bus","car","horse","knife","motorcycle","person","plant","skateboard","train","truck"]

# Argparser with dir input 
parser = argparse.ArgumentParser(description='VISDA structure')
parser.add_argument('--dir', type=str, default='', help='path to dataset')
parser.add_argument('--map', type=str, default='', help='path to map file')


args = parser.parse_args()

map_files = {}

# Read map file 
with open(args.map, 'r') as f: 
    for line in f: 
        file_name, id = line.rstrip().split()
        file_name = file_name.split('/')[-1]
        map_files[file_name] = int(id)


# Read directory args.dir recursively
for root, dirs, files in os.walk(args.dir):
    
    # Load file 
    for file in files:
        
        img_file = f"{file}"
        # img_file = img_file[2:]
        print(root, dirs, img_file)
        if img_file.endswith('.jpg') and 'trunk' in root:
            img_id = map_files[img_file]

            # Create folder
            if not os.path.exists(f"{args.dir}/{classes[img_id]}"):
                os.makedirs(f"{args.dir}/{classes[img_id]}")


            # Move file
            os.rename(f"{root}/{file}", f"{args.dir}/{classes[img_id]}/{file}")