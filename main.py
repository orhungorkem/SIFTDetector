import cv2
import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import cdist
import os



# Get fps of given video
def getFps(path):

    vidObj = cv2.VideoCapture(path)
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    vidObj.release()
    return int(fps)


# Given the path to video, the fps and the second of needed frame, it returns the frame in jpg format ( needed for testing and trials )
def saveFrame(path, fps, frame_no):

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    success = 1
    frame_count = 0


    while success:

        success, img = vidObj.read()
        
        if frame_count == frame_no:
            cv2.imwrite("frame"+str(frame_count)+".jpg",img)
            break
  
        # Catch the frames per second
        if count % fps == 0:
            frame_count = frame_count + 1
        
  
        count += 1
    
    




# Path: Path to the video to capture descriptors
# Fps: Fps of the video
# Interval: Array with two elements that indicate the start and end time of video to capture ([0,420] for first 7 min)
# No_of_descriptors: SIFT captures many descriptors most of which are unnecessary. This parameter determines the number of descriptors to capture with biggest blobs. 
# Can be reduced to some extent with efficiency concerns.
# Folder_to_save: Descriptors are saved to a subfolder under ./descriptors. Name of the subfolder should be given. 
# Function saves 3 files:
# * address.json: Mapping of descriptors to frames  ({"352":2} means descriptor in 352. row is the first descriptor of frame 2)
# * descriptors.npy: A 2d numpy array where each row is a descriptor (which is a 128 byte array). Each frame has no_of_descriptors rows in this array.
# * angles.npy: A 2d array that keeps principle angle of each keypoint in a frame in each row. 
# (Each row has no_of_descriptors elements since there are no_of_descriptors keypoints for each frame. And there are as many rows as the number of frames captured.)
# Ex. interval = [20,40] and no_of_descriptors = 150
# Then the frames between 20. and 40. seconds of the given video are analyzed. 
# descriptors.npy will have the shape (150*20, 128) since each row is a descriptor and total number of descriptors is 150*20
# angles.npy will have the shape (20,150) since each row is a frame and each descriptor is a column
def captureDescriptors(path, fps, interval, folder_to_save, no_of_descriptors=150):

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    success = 1
    start = interval[0]
    end = interval[1]
    detect = cv2.xfeatures2d.SIFT_create(no_of_descriptors)

    all_desc = None
    all_angles =[]
    for i in range(start):
        all_angles.append([])
    first = True
    rowcount = 0
    frame_address = {}  # the mapping from row of decriptors to the frame number
    frame_count = start  # we catch the frame by second

    while success:

        if (count / fps) >= end:
            break

        success, img = vidObj.read()

        if (count / fps) < start:
            count += 1
            continue
        
  
        # Catch the frames per second
        if count % fps == 0:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detect.detectAndCompute(img,None)
            angles = [int(key.angle) for key in keypoints]
            all_angles.append(angles)
            if first:
                all_desc = descriptors
                first = False
            else:
                all_desc = np.concatenate((all_desc, descriptors))

            frame_address[rowcount] = frame_count
            rowcount = rowcount + len(descriptors)
            frame_count = frame_count + 1
  
        count += 1
    
    if not os.path.exists("./descriptors/"+folder_to_save):
        os.mkdir("./descriptors/"+folder_to_save)

    np.save("./descriptors/"+folder_to_save+"/angles", all_angles)
    np.save("./descriptors/"+folder_to_save+"/descriptors", all_desc)
    with open('./descriptors/'+folder_to_save+'/address.json', 'w') as fp:
        json.dump(frame_address, fp)
    print("Features saved")
        




# Path: Path to the video to analyze
# Fps: Fps of the video
# Interval: Array with two elements that indicate the start and end time of video to analyze ([420,840] between 7. and 14. mins)
# No_of_descriptors: SIFT captures many descriptors most of which are unnecessary. This parameter determines the number of descriptors to capture with biggest blobs
# Desc: descriptors.npy which is obtained by captureDescriptors()
# Sq: address.json which is obtained by captureDescriptors()
# Ang: angles.npy which is obtained by captureDescriptors()
# Ratio: When a descriptor is compared to a set of descriptors, we call the most similar pair a "match". 
# To call it a "good match", we need that the distance of the match must me smaller than a ratio of the second best match. 
# If ratio = 0.7, distances of first two matches are d1 and d2, the match with distance of d1 is a good match if d1 < 0.7*d2. 
# We only count the good matches, thus ratio is an important parameter.
# Dumpfile: The file to write the matching results. (need to be a .csv)
# Function reads the given interval of the video, extracts the SIFT features of each frame, then compares the features with the ones in database. 
# For our case, the database is given with desc, sq, ang. This can be changed. With the comparison, match results are written to a .csv file.
def analyzeFrames(path, interval, desc, sq, ang, no_of_descriptors, fps, dumpfile, ratio = 0.75):

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    success = 1
    start = interval[0]
    end = interval[1]
    detect = cv2.xfeatures2d.SIFT_create(no_of_descriptors)
    first = True

    while success:

        if (count / fps) >= end:
            break

        success, img = vidObj.read()

        if (count / fps) < start:
            count += 1
            continue
        
  
        # Catch the frames per second
        if count % fps == 0:
            
            frame_no = int(count/fps)
            print(frame_no)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            new_keypoints, new_descriptors = detect.detectAndCompute(img,None)
            angles = [int(key.angle) for key in new_keypoints]
            d = np.array(cdist(new_descriptors, desc))
            
            matches, matched, glob_match = getMatchFromDistance(sq, d, ratio)
            startidx = 0
            for key, value in sq.items():
                if value == matched:
                    startidx = int(key)
                    break
            matched_ang1 = []
            matched_ang2 = []
            for m in glob_match:
                new_idx = m[0]
                old_idx = m[1]
                if old_idx>=startidx and old_idx <startidx + no_of_descriptors:
                    idx = old_idx - startidx
                    angle1 = angles[new_idx]
                    angle2 = ang[matched][idx]
                    matched_ang1.append(angle1)
                    matched_ang2.append(angle2)
            angle, _ = detectAngle(matched_ang1, matched_ang2) 
            writeMatches(frame_no, len(sq), matches, matched, angle, first, dumpfile)
            if first:
                first = False
  
        count += 1
    


    

# d: The distance matrix between descriptors of a frame and the set of descriptors in the database. 
# Shape of d is (n,m) if current frame has n descriptors and there are m descriptors in database. 
# d_ij = Distance between the ith descriptor of the frame and jth descriptor in the database. 
# Function returns 3 things: 
# * matches: An array that counts the number of matches between the current frame and each of the frames in database.
# * matched: argmax(matches) , the frame that is the best match of the current frame (test frame) 
# * glob_match: An array of tuples where each element (i,j) is a pair of indices of matched descriptors. 
# (i,j) means that ith descriptor of test frame is matched with jth descriptor in database. We get this to find relative angles.   
def getMatchFromDistance(sq, d, ratio):
    rows, _ = d.shape   
    matches = [0 for _ in range(len(sq))]
    indices = []
    glob_match = []
    for i in range(rows):
        row = d[i]
        min1, min2 = np.partition(row, 1)[0:2]
        if min1 < ratio*min2:
            # means this is a good match
            idx = np.where(row == min1)[0][0]
            indices.append(idx) 
            glob_match.append((i,idx))   
    for idx in indices:
        last = '0'
        for k in sq:
            if idx > int(k):
                last = k
                continue
            else:
                matched_square = sq[last]
                matches[matched_square] += 1   
                break
    matched = np.argmax(matches)
    return matches, matched, glob_match






# http://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html
# Gets two arrays of angles to compare. Arrays have one to one correspondence. That is, ith elements of both arrays belong to matched keypoints. 
# Difference between each corresponding pair of angles is calculated. 
# The most common difference is inferred to be the relative angle between test frame and matched database frame.
def detectAngle(angles1, angles2):

    counter = np.array([0 for i in range(360) ])
    for i in range(len(angles1)):
        diff = angles1[i] - angles2[i]
        if diff < 0:
            diff += 360
        counter[diff] += 1
    return np.argmax(counter), np.max(counter) 


    
# Matching results are written to a csv file. 
def writeMatches(frame_no, no_of_frames, matches, matched, angle, first, dumpfile):


    total_matches = sum(matches)
    max_match = matches[matched]

    if not first:
        df = pd.read_pickle(dumpfile)
        
    else:
        columns = ["Frame no","Matched Frame", "Angle" ,"Total Matches", "Max Match"]
        for i in range(no_of_frames):
            columns.append(i)
        df = pd.DataFrame(columns=columns)
    dic = {"Frame no": [frame_no], "Matched Frame": [matched], "Angle":[angle], "Total Matches":[total_matches], "Max Match":[max_match]}
    for i in range(no_of_frames):
        dic[i] = [matches[i]]
    df2 = pd.DataFrame(dic, index=[0])
    df = pd.concat([df, df2], sort=False)
    
    
    df.to_pickle(dumpfile)



# folder name of the run, will appear under matches directory
folder = "whitesquares"

# parameters of captureDescriptors()
train_video = "./videos/karolar_2.mov"
train_fps = 30
train_interval = [0,430]
train_descriptors = 150 

# parameters of analyzeFrames()
query_video = "./videos/karolar_2.mov"
query_fps = 30
query_interval = [430,1320]
query_descriptors = 150
ratio = 0.75


# make it false if the descriptors in the database are being used
train = True


test = False



if train:
    captureDescriptors(path = train_video,fps = train_fps, interval = train_interval, folder_to_save = folder, no_of_descriptors = train_descriptors)




if test:
    with open('./descriptors/'+folder+'/address.json', 'r') as fp:
        sq = json.load(fp)
    with open('./descriptors/'+folder+'/descriptors.npy', 'rb') as f:
        desc = np.load(f)
    with open('./descriptors/'+folder+'/angles.npy', 'rb') as f:
        ang = np.load(f,allow_pickle=True)   

    analyzeFrames(path = query_video, interval = query_interval, desc = desc, sq = sq, ang = ang, no_of_descriptors = query_descriptors, 
                    fps = query_fps, folder = './matches/'+folder+'.csv', ratio = ratio)

    df = pd.read_pickle("./matches/"+folder+".csv")
    df.to_csv("./matches/"+folder+".csv")



