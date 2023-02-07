from PIL import Image
from absl import app, flags
from concurrent import futures
import os 

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', "/tmp/", "Dir to convert images")


def resize_crop(image_name): 

	im = Image.open(image_name)
	width, height = im.size   # Get dimensions

	if width >= height: 
		width = int(width*1.0/height*256)
		height = 256
	else: 
		height = int(height*1.0/width*256)
		width = 256

	im = im.resize((width, height))

	# width, height = im.size	
	# new_width, new_height = 224, 224

	# left = (width - new_width)/2
	# top = (height - new_height)/2
	# right = (width + new_width)/2
	# bottom = (height + new_height)/2

	# # Crop the center of the image
	# im = im.crop((left, top, right, bottom))
	im.save(image_name)

def main(_):

	pool = futures.ThreadPoolExecutor(20)

	processes = []	
	for r, d, f in os.walk(FLAGS.dir):
		for file in f: 
			if file.endswith("png") or file.endswith("PNG") or file.endswith("jpg") or file.endswith("JPEG") or file.endswith("jpeg") or file.endswith("JPG"):
				process = pool.submit(resize_crop, r + "/" + file)
				processes.append(process)
	
	futures.wait(processes)



if __name__ == '__main__':
	app.run(main)