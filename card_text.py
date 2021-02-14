import urllib.request
import os
from os import path
import json
import time

CARD_FILEPATH = "card_info"
DICT_FILEPATH = "dicts"
api_url = "https://api.scryfall.com/cards/named?fuzzy="

def main():
  for filename in os.listdir(DICT_FILEPATH):
    haystack = {}
    with open(DICT_FILEPATH+"/"+filename) as json_file:
      haystack = json.load(json_file)
      for name in haystack:
        #storing card text
        if not path.exists(CARD_FILEPATH+"/"+name+".json"): #this check costs some efficiency, but limits our api requests
          url_ending = ""
          words = name.split()
          for word in words:
            url_ending+=word+"-"
          url_ending = url_ending[:-1]
          print(url_ending) #delete this if you don't like cluttering the terminal
          response = urllib.request.urlopen(api_url+url_ending)
          data = json.loads(response.read())
          output_path = os.path.join("card_info", name+".json")
          with open (output_path, "w") as f:
            json.dump(data, f)
            time.sleep(0.1)#api rules
        with open(CARD_FILEPATH+"/"+name+".json") as json_file:
          card_info = json.load(json_file)
          card_text = name + " " + card_info["oracle_text"]
        l = haystack.get(name, [])
        l.append(card_text)
        haystack[name] = l
    output_path = os.path.join(DICT_FILEPATH+"/"+filename)
    with open (output_path, "w") as f:
      json.dump(haystack, f)

if __name__ == "__main__":
  main()