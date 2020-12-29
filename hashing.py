from imutils import paths
from PIL import Image
import argparse
import time
import sys
import cv2
import os
import json
import imagehash
import numpy

REFERENCE_WIDTH = 265
REFERENCE_HEIGHT = 370

def dhash(image, hashSize=5):
	# resize the input image
    resized = cv2.resize(image, (hashSize + 2, hashSize))
    # compute the (relative) horizontal gradient between adjacent
	# column pixels
    diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
	help="dataset of images to hash")
args = vars(ap.parse_args())

#grab the paths to both the haystack and needle images 
print("[INFO] computing hashes for haystack...")
haystackPaths = list(paths.list_images(args["haystack"]))
# remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
	haystackPaths = [p.replace("\\", "") for p in haystackPaths]

# initialize the dictionary that will map the image hash to corresponding image,
# hashes, then start the timer
haystack = {}
start = time.time()

# loop over the haystack paths
for p in haystackPaths:

    # # load the image from disk
    # image = cv2.imread(p)
    # #resized = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))
    # cv2.imshow("current card", image)
    # cv2.waitKey(0)
    # name = input("enter card name: ")
    # # if the image is None then we could not load it from disk (so
    # # skip it)
    # if image is None:
    #     continue
    # # convert the image to grayscale and compute the hash
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = cv2.resize(image, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
    # imageHash = dhash(image)
    # #imageHash = cv2.img_hash_ImgHashBase.compute(image)

    image = cv2.imread(p)
    cv2.imshow("current card", image)
    cv2.waitKey(0)
    name = input("enter card name: ")
    imageHash = str(imagehash.average_hash(Image.open(p)))
    
    # update the haystack dictionary
    l = haystack.get(imageHash, [])
    l.append(name)
    haystack[imageHash] = l
    cv2.destroyAllWindows()

# show timing for hashing haystack images, then start computing the
# hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(
	len(haystack), time.time() - start))

s = input("input dict name: ")
output_path = os.path.join("dicts", s+".json")
with open (output_path, "w") as f:
    json.dump(haystack, f)

# print("[INFO] computing hashes for needles...")

# # loop over the needle paths
# for p in needlePaths:
# 	# load the image from disk
# 	image = cv2.imread(p)
# 	# if the image is None then we could not load it from disk (so
# 	# skip it)
# 	if image is None:
# 		continue
# 	# convert the image to grayscale and compute the hash
# 	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	imageHash = dhash(image)
# 	# grab all image paths that match the hash
# 	matchedPaths = haystack.get(imageHash, [])
# 	# loop over all matched paths
# 	for matchedPath in matchedPaths:
# 		# extract the subdirectory from the image path
# 		b = p.split(os.path.sep)[-2]
# 		# if the subdirectory exists in the base path for the needle
# 		# images, remove it
# 		if b in BASE_PATHS:
# 			BASE_PATHS.remove(b)

# # display directories to check
# print("[INFO] check the following directories...")
# # loop over each subdirectory and display it
# for b in BASE_PATHS:
# 	print("[INFO] {}".format(b))

