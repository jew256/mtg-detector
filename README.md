# Magic the Gathering Card Detector

## Purpose

This project is intended to provide a visual tool for playing the trading card game Magic: the Gathering with physical cards online with someone else.\
There are different forms of online magic already, but this project would allow personal collections of cards to be used, without the difficulty of reading cards from another's webcam feed. \
Instead of a fuzzy image of your opponent's board, this program creates a virtual playing board to easier view each card on their physical board. 

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

In its current state, this program only provides a one-way service where the host submits cards to the server and someone else can remote into the server to view\
For both players to view each other's board, in the current state each player must have an instance of the program running locally.\
To host, an additional tunneling program program is required to project the local server.\
For this project, localtunnel is recommended. Ngrok is not recommended because of its limit to connections often maxes out when adding new cards to the board.

1. install localtunnel `npm install -g localtunnel`
2. host on port 5000 `lt --port 5000`
3. run card_detector.py to start the local server `pythyon3 card_detector.py`
4. copy the url from localtunnel and send to your opponent for them to view your board

## Development History

### Card Detection

At first I had hoped to find a card detection algorithm online to adapt to my purposes, but I was unable to find anything suitable for what I wanted. 
However, this program's card detection algorithm takes inspiration from [Timo Ikonen's card detector](https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/), but is less involved in pre-processing to maintain speed for the user experience.\
In addition, the scope of this project assumes more ideal conditions than he accounts for.\
The Hashing approach is similar, but I ran into the issue of compiling the necessary references to run the algorithm on any card wanted.\

#### Dec 27, 2020: Finding Regions of Interest (ROI)

I first started with determining the regions of interest from the camera frame. No frame will be perfect, so the first step is to separate just the card from the image before hashing.

openCV is used for all operations on the frame and a few methods for finding the bounding rectangle (because every card is a rectangle), adjusting for rotation, and setting bounds for the size of the card all serve to separate the card from the frame.

Note: the program is better suited to handle slight counter-clockwise rotations, but not clockwise rotations

![finding_roi](https://github.com/jew256/mtg-detector/blob/master/README_images/finding_card.png?raw=true)

### Web Crawler

### Hosting on a Server

## Current State

## Future Development
