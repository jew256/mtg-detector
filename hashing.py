from imutils import paths
from PIL import Image
from parsel import Selector
from scrapy.crawler import CrawlerProcess
import argparse
import time
import sys
import cv2
import os
import json
import imagehash
import numpy as np
import scrapy
import urllib.request

REFERENCE_WIDTH = 265
REFERENCE_HEIGHT = 370

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--set_name", required=True,
	help="name of set to be parsed")
ap.add_argument("-b", "--url", required=True,
	help="url to crawl")
args = vars(ap.parse_args())

set_dict = {}
set_name = args["set_name"]
set_url = args["url"]
api_url = "https://api.scryfall.com/cards/named?fuzzy="

def url_to_image(url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image

def hash(title, image_url):
    image = url_to_image(image_url)
    im_pil = Image.fromarray(image)
    imageHash = str(imagehash.average_hash(im_pil))
    return imageHash

def process_names(names):
    while names[0] == '':
        names.pop(0)
    set_name = names.pop(0)
    while names[0] == '':
        names.pop(0)
    return names

def process_links(links):
    image = url_to_image(links[0])
    while image.shape[1] != REFERENCE_WIDTH:
        links.pop(0)
        image = url_to_image(links[0])
    return links

class SetSpider(scrapy.Spider):
    start_urls = [args["url"]]
    name = args["set_name"]
    def parse(self, response):
        
        names = response.css('img::attr(alt)').extract()
        links = response.css('img::attr(src)').extract()
        names = process_names(names)
        links = process_links(links)
        for i in range(len(names)):
            title = names[i]
            title = title.replace("\u2019", "")
            title = title.replace("'", "")
            link = links[i]
            #adding image hash
            imageHash = hash(title, link)
            l = set_dict.get(title, [])
            l.append(imageHash)
            set_dict[title] = l
            
            
def main():
    print("[INFO] computing hashes for images...")
    start = time.time()
    process = CrawlerProcess()
    process.crawl(SetSpider)
    process.start()
    print("[INFO] processed {} images in {:.2f} seconds".format(
        len(set_dict), time.time() - start))

    output_path = os.path.join("dicts", set_name+".json")
    with open (output_path, "w") as f:
        json.dump(set_dict, f)

if __name__ == "__main__":
    main()