from PIL import Image
import imagehash

hash = imagehash.average_hash(Image.open("input/20201228_202201.jpg"))
otherhash = imagehash.average_hash(Image.open("modern horizons/en_x0FNqXC9Y8.png"))

print(hash-otherhash)
