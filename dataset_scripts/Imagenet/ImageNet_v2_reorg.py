# from PIL import Image
from absl import app, flags
from concurrent import futures
import os 
import json 

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', "/tmp/", "Dir to convert images")
flags.DEFINE_string('info', "/tmp/", "json file")

def main(_):


	file_map = {}
	with open(FLAGS.info, "r" ) as file: 
		
		json_array = json.load(file)
		for i, line in enumerate(json_array): 
			# print(line)
			file_map[str(line[0])] = line[1]			

	for r, d, f in os.walk(FLAGS.dir):
		dir_name = r.split("/")[-1]
		if dir_name in file_map:
			os.rename(r, "/".join(r.split("/")[:-1]) + "/" + file_map[dir_name])


if __name__ == '__main__':
	app.run(main)
