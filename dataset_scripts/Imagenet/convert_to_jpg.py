from PIL import Image
from absl import app, flags
from concurrent import futures
import os 

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', "/tmp/", "Dir to convert images")


def jpg_to_png(image_name): 

	im = Image.open(image_name)
	im.save(f"{image_name[:-4]}.jpg")

def main(_):

	pool = futures.ThreadPoolExecutor(20)

	processes = []	
	for r, d, f in os.walk(FLAGS.dir):
		for file in f: 
			if file.endswith("png") or file.endswith("PNG"):
				process = pool.submit(jpg_to_png, r + "/" + file)
				processes.append(process)
	
	futures.wait(processes)



if __name__ == '__main__':
	app.run(main)