import os
import csv
audio_dir = r"C:/Users/andrew/Downloads/ml_stuttering/clips/stuttering-clips/clips"
csvFile = r"detection.csv"

def insertIntoCSV(data, c=csvFile):
    with open(c, 'a', newline='') as cfile:
        writer = csv.writer(cfile)
        writer.writerow(data)
        cfile.close()

def extractClip(filename):
    for filename in os.listdir(audio_dir):
        f = filename.strip().split("_")
        b = True
        data = [f[0],f[1],f[2][:-4], b]
        insertIntoCSV(data)

# extractClip(audio_dir)