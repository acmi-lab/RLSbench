from PIL import Image
from absl import app, flags
from concurrent import futures
import os 

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', "/tmp/", "Dir to convert images")
flags.DEFINE_string('newDir', "/tmp/", "Dir to convert images")
flags.DEFINE_string('csv', "/tmp/", "CSV file")

def main(_):


	file_map = {}
	with open(FLAGS.csv, "r" ) as file: 
		file.readline()
		for line in file: 
			file_name, file_class = line.split(" ")[0].split(",")
			file_map[file_name + ".JPEG"] = file_class

			print(file_name +  ".JPEG", file_class)
			

	for r, d, f in os.walk(FLAGS.dir):
		for file in f: 
			if file.endswith("jpg") or file.endswith("JPEG") or file.endswith("jpeg") or file.endswith("JPG"):
				file_name =  (r + "/" + file).split("/")[-1]
				if not os.path.isdir(FLAGS.newDir + "/" + file_map[file_name]): 
					os.makedirs(FLAGS.newDir + "/" + file_map[file_name])

				os.rename(r + "/" + file, FLAGS.newDir + "/" + file_map[file_name] + "/" + file_name )


if __name__ == '__main__':
	app.run(main)