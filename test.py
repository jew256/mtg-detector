from PIL import Image
import imagehash

hash = imagehash.average_hash(Image.open("input/spore_frog.jpg"))
otherhash = imagehash.average_hash(Image.open("modern horizons/en_YDoMxhGqUf.png"))

print(otherhash)
