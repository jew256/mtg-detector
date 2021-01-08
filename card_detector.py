from PIL import Image
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
from flask_socketio import send, emit
#from flask_session import Session
from os import path
import numpy as np
import cv2
import imagehash
import json
import os
import urllib.request
import time
import sys
import base64
import itertools

#TODO: make card images clickable to view information - a task not suitable for flask?
#TODO: make actual virtual board for cards to sit on: able to drag them around to reposition
#TODO: implement life counter
#TODO: instead of one host. make clients able to log into rooms on server to play games
#TODO: more than 2 clients at once. display up to 4 playing boards at the same time

ASPECT_THRESHOLD = 0.7
AREA_LOWER_THRESHOLD = 0.03
AREA_UPPER_THRESHOLD = 0.98
HASH_TOLERANCE = 8
REFERENCE_WIDTH = 265
REFERENCE_HEIGHT = 370
INPUT_FILEPATH = "input"
DICT_FILEPATH = "dicts"
REFERENCE_FILEPATH = "modern horizons"
CARD_FILEPATH = "card_info"
CARD_BACK = "https://media.magic.wizards.com/image_legacy_migration/magic/images/mtgcom/fcpics/making/mr224_back.jpg"

api_url = "https://api.scryfall.com/cards/named?fuzzy="
current_board = []
# current_card = ""
#global current_card #placeholder img

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Add Card':
            stringData=add_card()
            if stringData != "":
                return render_template('index.html', board=stringData)
        elif request.form['submit_button'] == 'Delete Card':
            text = request.form['text']
            stringData=delete_card(text)
            if stringData != "":
                return render_template('index.html', board=stringData)
    return render_template('index.html')

def add_card():
    f = open("current_card.txt")
    current_card = f.read()
    f.close()
    if current_card == "":
        return ""
    current_board.append(current_card)
    board = create_board(current_board)
    r, cnt = cv2.imencode('.jpg',board)
    stringData = base64.b64encode(cnt).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData
    return stringData
    
def delete_card(card_name):
    for filename in os.listdir(CARD_FILEPATH):
        if filename == card_name + ".json":
            with open(CARD_FILEPATH+"/"+filename) as json_file:
                card_info = json.load(json_file)
            image = card_info["image_uris"]["png"]
            if image in current_board:
                current_board.remove(image)            
                print("removed: " + card_name)
    if len(current_board) > 0:
        board = create_board(current_board)
        r, cnt = cv2.imencode('.jpg',board)
        stringData = base64.b64encode(cnt).decode('utf-8')
        b64_src = 'data:image/jpg;base64,'
        stringData = b64_src + stringData
        return stringData
    return ""

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (h_min, int(im.shape[1] * h_min / im.shape[0])), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def create_board(current_board):
    board_images = []
    for url in current_board:
        board_images.append(url_to_image(url))
    margin = 20 #Margin between pictures in pixels
    w = 5 # Width of the matrix (nb of images)
    h = int(len(board_images)/w) # Height of the matrix (nb of images)
    if len(board_images)%w != 0:
        h+=1
    n = w*h

    #Define the margins in x and y directions
    m_x = margin
    m_y = margin

    #Define the shape of the image to be replicated (all images should have the same shape)
    img_h, img_w, img_c = board_images[0].shape


    #Size of the full size image
    mat_x = img_w * w + m_x * (w - 1)
    mat_y = img_h * h + m_y * (h - 1)

    #Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
    imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
    imgmatrix.fill(255)

    #Prepare an iterable with the right dimensions
    positions = itertools.product(range(h), range(w))

    for (y_i, x_i), card in zip(positions, board_images):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = card

    resized = cv2.resize(imgmatrix, (mat_x//3,mat_y//3), interpolation = cv2.INTER_AREA)
    return resized

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            image, roi = find_cards(frame)
            urls = analyze_ROI(roi)
            #urls.append("https://c1.scryfall.com/file/scryfall-cards/png/front/8/3/83298c8a-02c4-4ada-9a41-4b973bb58ac6.png?1562201132")
            if urls:
                with app.test_request_context('/'):
                    socketio.emit('data', urls[0])

                f = open("current_card.txt", "w")
                f.truncate(0)
                f.write(urls[0])
                f.close()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def rotate_image(image, angle):
#     # Grab the dimensions of the image and then determine the center
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w / 2, h / 2)

#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])

#     # Compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))

#     # Adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY

#     # Perform the actual rotation and return the image
#     return cv2.warpAffine(image, M, (nW, nH))



def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))

def find_cards(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 0)

    #edges = cv2.Canny(thresh,100,200)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    roi = []
    ROI_number = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015*peri, True)

        if len(approx) == 4:
            rect = cv2.minAreaRect(c)
            (x, y), (width, height), angle = rect
            # aspect_ratio = min(width, height) / max(width, height)
            area = width*height
            frame_area = frame.shape[0]*frame.shape[1]
            if area/frame_area > AREA_LOWER_THRESHOLD and area/frame_area < AREA_UPPER_THRESHOLD:
                #print(aspect_ratio)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame,[box],0,(0,0,255),2)
                ROI_number += 1
                
                # get width and height of the detected rectangle
                width = int(rect[1][0])
                height = int(rect[1][1])

                src_pts = box.astype("float32")
                # coordinate of the points in box points after the rectangle has been
                # straightened
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                warped = cv2.warpPerspective(frame, M, (width, height))
                #cv2.imshow("warped", warped)
                #cv2.waitKey(0)
                roi.append(warped)
                # resized = cv2.resize(warped, (REFERENCE_WIDTH, REFERENCE_HEIGHT))qq
                #cv2.imshow("ROI" + str(ROI_number), resized)

            #cv2.drawContours(frame, [c], 0, (36, 255, 12), 3)
    return frame, roi

def analyze_ROI(roi):
    roi_index = 0
    cards_this_frame = []
    urls = []
    for r in roi: 

        # #tesseract attempt
        # resized = cv2.resize(r, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
        # crop_img = resized[0:50, 0: REFERENCE_WIDTH-50]
        # print(pytesseract.image_to_string(crop_img))
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)

        #current = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        current = cv2.resize(r, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
        # cv2.imshow("current", current)
        # cv2.waitKey(0)
        
        im_pil = Image.fromarray(current)
        imageHash = imagehash.average_hash(im_pil)

        min_dif = 100
        closest_card = ""

        for filename in os.listdir(DICT_FILEPATH):
            haystack = {}
            with open(DICT_FILEPATH+"/"+filename) as json_file:
                haystack = json.load(json_file)
            for hash in haystack:
                difference = abs(imageHash-imagehash.hex_to_hash(hash))
                if difference < min_dif:
                    min_dif = difference
                    closest_card = str(haystack[hash][0])
            
        if min_dif < HASH_TOLERANCE:
            # print(haystack[hash])
            if closest_card not in cards_this_frame:
                cards_this_frame.append(closest_card)
                print("this card is: " + closest_card)
                x = 10
                y = int(REFERENCE_HEIGHT/2)
                current = cv2.rectangle(current,(x, y-20),(REFERENCE_WIDTH-x,y+20),(255,255,255),-1)
                current = cv2.putText(current, closest_card, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA) 
                #cv2.imshow("card_"+str(roi_index), current)

                #cv2.moveWindow("card_"+str(roi_index), roi_index*REFERENCE_WIDTH, 0)

                #manually check this card is correct
                confirmation = input("is this card " + closest_card + "? y/n \t")
                #confirmation = "y"

                if confirmation == 'y':
                    #find data about card (must not happen more than 10 times per second)
                    if not path.exists(CARD_FILEPATH+"/"+closest_card+".json"):
                        words = closest_card.split()
                        url_ending = ""
                        for word in words:
                            url_ending+=word+"-"
                        url_ending = url_ending[:-1]
                        response = urllib.request.urlopen(api_url+url_ending)
                        data = json.loads(response.read())
                        output_path = os.path.join("card_info", closest_card+".json")
                        with open (output_path, "w") as f:
                            json.dump(data, f)
                        time.sleep(0.1)
                    urls.append(parse_card_info(closest_card))
                    #exit()

                
        roi_index+=1
    #cv2.destroyAllWindows()
    return urls

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def parse_card_info(closest_card):
    #card_info = {}
    with open(CARD_FILEPATH+"/"+closest_card+".json") as json_file:
        card_info = json.load(json_file)

    #display card
    image = url_to_image(card_info["image_uris"]["png"])
    time.sleep(0.1) #delay per api rules
    #cv2.imshow("card", image)    

    #return color
    colors = card_info["colors"]
    print("this card's colors are: " + str(colors))

    #return cost
    cost = card_info["mana_cost"]
    print("this card's cost is: " + str(cost))

    #return cmc
    cmc = card_info["cmc"]
    print("this card's cmc is: " + str(cmc))

    #return oracle text
    effect = card_info["oracle_text"]
    print("text: " + str(effect))

    return card_info["image_uris"]["png"]


def video_capture():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # operations on the frame
        image, roi = find_cards(frame)
        analyze_ROI(roi)
        # Display the resulting frame
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def image_input():
    for filename in os.listdir(INPUT_FILEPATH):
        img = cv2.imread(os.path.join(INPUT_FILEPATH,filename))
        resized_original = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
        #cv2.imshow('original', resized_original)
        image, roi = find_cards(img)
        resized = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))
        # cv2.imshow('frame', resized)
        analyze_ROI(roi)
        cv2.imshow('frame',image)
        cv2.waitKey(0)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def server_input():
    frame = sys.argv[1]
    frame = np.float32(frame)
    image, roi = find_cards(frame)
    analyze_ROI(roi)

def main():
    video_capture()
    #image_input()
    #server_input()

if __name__ == "__main__":
    socketio.run(app)
