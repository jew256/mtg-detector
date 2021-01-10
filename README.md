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
To host, an additional tunneling program is required to project the local server.\
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

#### Dec 28, 2020: Detection by hashing

After separating the card from the image, a [hash](https://pypi.org/project/ImageHash/) is created and the difference between this hash and every card in the reference pool is taken. \
If the minimum difference is less than the tolerance for a match, then the corresponding reference image is assumed to be the matching card.\
In the image below, the program is reading a local image and comparing the card's hash to that of a few reference images.

![hashing_detection](https://github.com/jew256/mtg-detector/blob/master/README_images/hashing_detection.png?raw=true)

#### Dec 29, 2020: Video Input

After detecting the card from a single image, the next step is detecting a card in each frame of a video feed.\
openCV is used to open the camera and read in each frame. The same algorithm is applied to each individual frame, relating it to reference images.

[![Watch the video](https://img.youtube.com/vi/vKajeu5hVDo/maxresdefault.jpg)](https://youtu.be/vKajeu5hVDo)

### Web Crawler

#### Dec 30, 2020: Building Set Dictionaries

Manually inputing every card for the reference set is unreasonable, so I built a web crawler to store every card's hash and name in a dictionary for every published set.\
This separate program builds a dictionary for these sets, and must be called for every set of cards required to be added.\
This approach also helps efficiency because instead of hashing every image in the reference set, those hashes are already created and can be referenced.\
The below image is detecting the card "Timberland Guide" with a reference set of "Modern Horizons" and "Iconic Masters"

[![Watch the video](https://img.youtube.com/vi/9oSt3JxGmKE/maxresdefault.jpg)](https://youtu.be/9oSt3JxGmKE)

#### Dec 30, 2020: Recognizing Multiple Cards

At first I believed that the program would take a frame from a playing board and build the entire virtual board every frame, so there would be no input from the user.\
Thus, I developed multiple-card recognition in, but later realized that this approach would be too computationally intense, as well as unreasonable for extremely cluttered boards.\
In the current state, the program requires the user to place their card underneath a camera, then add the card to their board.

![2_cards](https://github.com/jew256/mtg-detector/blob/master/README_images/2_cards.png?raw=true)

### API Integration

#### Jan 1, 2021: Acquiring Card Info

Once the card is detected, I wanted to provide the user with any information about the card that they hoped for.\
Thus, I integrated the program with the [Scryfall API](https://scryfall.com/docs/api) to request any information about the card.\
In the current state, the client cannot view this information, but I plan to provide it in a future version with card interaction.

![api_integration](https://github.com/jew256/mtg-detector/blob/master/README_images/api_integration.png?raw=true)

### Hosting on a Server

#### Jan 5, 2021: Hosting with node.js

I have no previous experience with hosting a server, so I first attempted hosting the project with [node.js](https://nodejs.org/en/)\
This approach was somewhat successful, but implementing the python program to dynamically update the board and current card proved challenging.\
However, in the video below, once the card was confirmed from the server side, the card would be added to the virtual board, shown below the life feed.

[![Watch the video](https://img.youtube.com/vi/a_LgCiw03zk/maxresdefault.jpg)](https://youtu.be/a_LgCiw03zk)

#### Jan 8, 2021: Shifting to Flask. Adding Add/Delete Options

I decided that [flask](https://flask.palletsprojects.com/en/1.1.x/) would be more suitable for my purposes, as I am dealing with the images and variables directly from the python script.\
In addition to the virtual board, I added a preview section of the card and add/delete buttons to manage the board state.\
This is closer to the future goal of having each player build their board from a connection to the separate server.\
The current board currently has no interacction, so a text field is used instead to delete particular cards from the board.
Note: in the video a number of false positives are found and there are multiple instances of the found card.\
These issues are a product of two factors, the lighting and hosting on flask.\
The accuracy of the detection relies on the images taken being close to the lighting of the reference images, so in this video the lighting was different and thus the accuracy diminished.\
As for the multiple instances of the card, I believe that the frames where the card is present on the board are stored and waiting in queue to be processed.\
Thus, while the first frame displays the found card, there are still a few more frames with the card yet to be processed.

[![Watch the video](https://img.youtube.com/vi/wv89BcE4I4I/maxresdefault.jpg)](https://youtu.be/wv89BcE4I4I)

## Future Development

### Board Interaction

I hope to have an interactive board for both players to use, so the experience is more fluid.\
A few features of this interactive board are:
    * Dragging the card images along a virtual board to rearrange them
    * Clicking on a particular card to have a larger view and possibly some information about the card
    * Displaying both boards with player names side-by-side so that both players can view their board as well as their opponent's

### Game Amenities

As this program is meant to allow players to play a game of Magic online, I hope to implement some of the features of a normal game.\
These features include: 
    * A life counter
    * Different sided die
    * Counters for card effects

### Multi-Player Support

### Visual Improvements

### Deep Learning for Card Detection 
