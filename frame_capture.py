
# Program To Read video
# and Extract Frames
import cv2
  
# Function to extract frames
def FrameCapture(path):
      
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
  
        # Saves the frames with frame-count
        if count % 15 == 0:
            cv2.imwrite("./frames2/frame%d.jpg" % int(count/15), image)
  
        count += 1
        


FrameCapture("siftvid.mp4")  


