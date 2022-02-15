import cv2
import numpy as np
import pandas as pd
import math
import json
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import time


# TODO
# * knn de k parametresiyle oyna
# dbscan bak
# eşlenen imageların princial orientationlarını da almalıyız. buna bak

def getFps(path):

    vidObj = cv2.VideoCapture(path)
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    vidObj.release()
    return int(fps)



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
    
    


# given video, fps, interval, it returns the descriptors from frames, the mapping of frames, 
# and the angle of each keypoint by a referance 
# interval is an array of start and end time in seconds ([0,420] for first 7 min)
def captureDescriptors(path, fps, interval, no_of_descriptors, folder_to_save):

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
    
    np.save("./"+folder_to_save+"/angles", all_angles)
    np.save("./"+folder_to_save+"/descriptors", all_desc)
    with open('./'+folder_to_save+'/address.json', 'w') as fp:
        json.dump(frame_address, fp)
    print("Features saved")
        





def analyzeFrames(path, interval, desc, sq, ang, no_of_descriptors, fps, dumpfile):

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
            
            matches, matched, glob_match = getMatchFromDistance(sq, d)
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
            angle, _ = detectAngle(matched_ang1, matched_ang2)  #second parameter may be necessary to show  success prob. of prediction
            writeMatches(frame_no, len(sq), matches, matched, angle, first, dumpfile)
            if first:
                first = False
  
        count += 1
    


    


def getMatchFromDistance(sq, d):
    rows, _ = d.shape   
    matches = [0 for _ in range(len(sq))]
    indices = []
    glob_match = []
    for i in range(rows):
        row = d[i]
        min1, min2 = np.partition(row, 1)[0:2]
        if min1 < 0.75*min2:
            # means this is a good match
            idx = np.where(row == min1)[0][0]
            indices.append(idx) #keeps the good match idnices, these will be mapped to squares
            glob_match.append((i,idx))   # we get this because we want to know the index in all_desc that corresponds to index in new_desc
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
def detectAngle(angles1, angles2):

    counter = np.array([0 for i in range(360) ])
    for i in range(len(angles1)):
        diff = angles1[i] - angles2[i]
        if diff < 0:
            diff += 360
        counter[diff] += 1
    return np.argmax(counter), np.max(counter) 


    

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




folder = "train_long"
train = True

if train:
    captureDescriptors("./videos/karolar_2.mov",30,[0,1350],150, folder_to_save = folder)

test = True



with open('./'+folder+'/address.json', 'r') as fp:
    sq = json.load(fp)

with open('./'+folder+'/descriptors.npy', 'rb') as f:
    desc = np.load(f)
with open('./'+folder+'/angles.npy', 'rb') as f:
    ang = np.load(f,allow_pickle=True)   #keeps the angles of keypoints for each saved frame


if test:
    analyzeFrames("./videos/query.mov",[0,58],desc,sq,ang,150,30,'./'+folder+'/matches.csv')

    df = pd.read_pickle("./"+folder+"/matches.csv")
    df.to_csv("./"+folder+"/matches.csv")




# açıyı tam float tutarlı alır mıyız