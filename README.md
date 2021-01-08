# Magic the Gathering Card Detector

## Set up

### Download and Setup

1. Download the entire repository either by zip or cloning
2. Check what sets your cards are from and see if there is a .json in the "dicts" folder for every set in your pool of cards
3. If additional sets are required, see "adding new sets" for information on addition more cards to the recognition pool
4. A webcam is required for this project. It is recommended to have an external webcam to be aimed down towards a flat surface
5. For best recognition, aim the webcam at a bright, flat surface to maximize the contrast between the border of the card and the background

### Adding new sets

A note on adding sets: "why not just add every set in existence so you don't have to worry about adding more?"\
The recognition algorithm depends on hashing the reference images in each set and comparing it to hashes of the input.\
There are many magic cards with unique art, so the more cards added will increase the likelihood of false positives.\
Unless an excellent camera is used, the hash of your input could be even more similar to another card than its actual reference.

1. When adding new sets, we will be working with "hashing.py" to crawl the webpages of Wizards of the Coast for individual sets of cards
2. There are two arguments for running hashing.py, the name of the set and the url to the webpage of each set
3. The name of the set is not entirely necessary, but its acts as a descriptor when removing unneeded sets in the future
4. The url for the set can be founded by searching online by searching for "[name of set] card list" and clicking the link by Wizards of the Coast
5. run hashing.py with the above two arguments to add all cards of that set to the reference pool. nothing else is required
    * example: python3 hashing.py -a "Modern Horizons" -b "https://magic.wizards.com/en/products/modernhorizons/cards" \
    will add the "Modern Horizons.json" to the dicts folder, and all cards from the set will be added to the reference pool


### Hosting

## Development History

## Current State

## Future Development
