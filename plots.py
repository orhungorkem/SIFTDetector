from matplotlib import pyplot as plt
import numpy as pd
import pandas as pd




def plotResults(frame, detector = 'sift',loss = 'l2', distance = True, matches = False, flann = False, bfknn = False):

    if flann: 
        address = "./data/"+detector+"_detection_"+loss+"_frame"+str(frame)+"_flann.csv"
        dist_address = "./plots/distances/"+detector+"_detection_"+loss+"_frame"+str(frame)+"_flann.jpg"
        match_address = "./plots/matches/"+detector+"_detection_"+loss+"_frame"+str(frame)+"_flann.jpg"
    elif bfknn:
        address = "./data/"+detector+"_detection_"+loss+"_frame"+str(frame)+"_bfknn.csv"
        dist_address = "./plots/distances/"+detector+"_detection_"+loss+"_frame"+str(frame)+"_bfknn.jpg"
        match_address = "./plots/matches/"+detector+"_detection_"+loss+"_frame"+str(frame)+"_bfknn.jpg"

    else:
        address = "./data/"+detector+"_detection_"+loss+"_frame"+str(frame)+".csv"
        dist_address = "./plots/distances/"+detector+"_detection_"+loss+"_frame"+str(frame)+".jpg"
        match_address = "./plots/matches/"+detector+"_detection_"+loss+"_frame"+str(frame)+".jpg"

    df = pd.read_csv(address)
    
    if distance:
        fig = plt.figure()
        plt.plot(df["Frame"],df["Distance"],marker = 'o')
        plt.xlabel("Frame no")
        plt.ylabel("Distance")
        plt.savefig(dist_address)
    if matches:
        fig = plt.figure()
        plt.plot(df["Frame"],df["Number of matches"],marker = 'o')
        plt.xlabel("Frame no")
        plt.ylabel("No of matches")
        plt.ylim([0,1000])
        plt.savefig(match_address)

for i in [1,2,3,4,5,6]:
    plotResults(i, bfknn = True, distance= False, matches= True)