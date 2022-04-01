# Color detection and segmentation. 
#1.​ Capture and store the background frame.[ This will be done for some seconds ] 
#2.​ Detect the red colored cloth using color detection and segmentation algorithm.
#3. Segment out the red colored cloth by generating a mask. [ used in code ]
#4. Generate the final augmented output to create a m
import cv2
import time
import numpy as np

#To save the output in a file output.avi
#FourCC is a 4-byte code used to specify the video codec. 
#The list of available codes can be found in fourcc.org. It is platform dependent. 
#FourCC code is passed as `cv.VideoWriter_fourcc('M','J','P','G')or cv.VideoWriter_fourcc(*'MJPG')` for MJPEG. 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#Starting the webcam
cap = cv2.VideoCapture(0)

#Allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

#We need to have a video that has some seconds dedicated to the background frame so that it could easily save the background image. 
#Capturing background for 60 frames
for i in range(60):
    ret, bg = cap.read()
#Flipping the background because camera captures inverted image
bg = np.flip(bg, axis=1)

#Reading the captured frame until the camera is open
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #Flipping the image for consistency
    img = np.flip(img, axis=1)

    #As we capture frames we are also capturing the colors in those frames.
    #we need to convert the images from BGR (Blue Green Red) to HSV (Hue, Saturation, Value).
    #We need to do this so that we can detect the red color more efficiently. 

    #Converting the color from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #1.Hue: This channel encodes color information.
    # Hue can be thought of as an angle where 0 degree corresponds to the red color, 
    # 120 degrees corresponds to the green color, and 240 degrees corresponds to the blue.

    #2. Saturation: This channel encodes the intensity/purity of color.
    # For example, pink is less saturated than red.   
    
    #3. Value: This channel encodes the brightness of color. 
    # Shading and gloss components of an image appear in this channel reading the videocapture video.  

    # We have to create masks which will check for the colors in the specified range and then mask it with the background image.
    # We'll be creating 2 different masks which will help us detect the colors in that given range. 

    #Generating mask to detect red colour
    #These values can also be changed as per the color
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255,255])
    mask_1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_1 = mask_1 + mask_2

    #We'll be adding the diluting effect to the image in the video. 
    #MORPH_OPEN and MORPH_DILATE are the two types of effects.

    #MORPH_OPEN: An opening is an erosion followed by a dilation.
    #Performing an opening operation allows us to remove small blobs from an image: 
    #first an erosion is applied to remove the small blobs, then a dilation is applied to regrow the size of the original object.

    #MORPH_DILATE: a dilation will grow the foreground pixels.
    #Dilations increase the size of foreground objects and are especially useful for joining broken parts of an image together.

    #Open and expand the image where there is mask 1 (color)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    #Selecting only the part that does not have mask one and saving in mask 2
    #We need to create a mask to segment out the red color from the frame
    mask_2 = cv2.bitwise_not(mask_1)

    #we need to create 2 resolutions. 
    #First one would be an image without color red 
    #Second one would be the background from the background image we captured earlier just for the parts where red color was (mask 1). 

    #Keeping only the part of the images without the red color 
    #(or any other color you may choose)
    res_1 = cv2.bitwise_and(img, img, mask=mask_2)

    #Keeping only the part of the images with the red color
    #(or any other color you may choose)
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    #Generating the final output by merging res_1 and res_2
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)
    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)


cap.release()
out.release()
cv2.destroyAllWindows()
