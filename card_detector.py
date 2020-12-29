from PIL import Image
import numpy as np
import cv2
import imagehash
import json
import os
import pytesseract 

ASPECT_THRESHOLD = 0.7
AREA_LOWER_THRESHOLD = 0.1
AREA_UPPER_THRESHOLD = 0.99
HASH_TOLERANCE = 8
REFERENCE_WIDTH = 265
REFERENCE_HEIGHT = 370
INPUT_FILEPATH = "input"
DICT_FILEPATH = "dicts"
REFERENCE_FILEPATH = "modern horizons"

def rotate_image(image, angle):
    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

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
    ret, thresh = cv2.threshold(gray, 50, 255, 0)

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
                roi.append(warped)
                resized = cv2.resize(warped, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
                #cv2.imshow("ROI" + str(ROI_number), resized)

            #cv2.drawContours(frame, [c], 0, (36, 255, 12), 3)
    return frame, roi

def dhash(image, hashSize=5):
	# resize the input image
	resized = cv2.resize(image, (hashSize + 2, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def analyze_ROI(roi):
    for r in roi: 

        # #tesseract attempt
        # resized = cv2.resize(r, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
        # crop_img = resized[0:50, 0: REFERENCE_WIDTH-50]
        # print(pytesseract.image_to_string(crop_img))
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)

        current = cv2.cvtColor(r, cv2.COLOR_BGR2BGRA)
        current = cv2.resize(current, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
        # cv2.imshow("current", current)
        # cv2.waitKey(0)
        
        im_pil = Image.fromarray(current)
        imageHash = imagehash.average_hash(im_pil)
        # for filename in os.listdir(REFERENCE_FILEPATH):
        #     # img = cv2.imread(os.path.join(REFERENCE_FILEPATH,filename))
        #     otherhash = imagehash.average_hash(Image.open(REFERENCE_FILEPATH+"/"+filename))
        #     #print(hash-otherhash)
        #     if hash-otherhash < 7:
        #         img = cv2.imread(os.path.join(REFERENCE_FILEPATH,filename))
        #         cv2.imshow("found", img)
        #         cv2.imshow("roi", current)
        #         cv2.waitKey(0)

        #imageHash = dhash(current)
        # imageHash = cv2.img_hash.averageHash(current)
        # reference = cv2.imread("modern horizons/en_YqZ34vWHL8.png")
        # referece = cv2.cvtColor(reference, cv2.COLOR_BGR2BGRA)
        # print(imageHash-cv2.img_hash.averageHash(referece))

        haystack = {}

        with open(DICT_FILEPATH+"/modern_horizons.json") as json_file:
            haystack = json.load(json_file)

        min_dif = 100
        closest_card = ""
        
        for hash in haystack:
            difference = abs(imageHash-imagehash.hex_to_hash(hash))
            if difference < min_dif:
                min_dif = difference
                closest_card = str(haystack[hash])

            #print(difference)
            
        if min_dif < HASH_TOLERANCE:
                # print(haystack[hash])
                print("this card is: " + closest_card)
                cv2.imshow("card", current)
                cv2.waitKey(0)
        # else:
        #     print("no matches found. closest match: " + closest_card + " with a difference of: " + str(min_dif))

        #print()

def video_capture():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # operations on the frame
        image, roi = find_cards(frame)

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
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def main():
    #video_capture()
    image_input()

if __name__ == "__main__":
    main()