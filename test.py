import pandas as pd
import numpy as np
import shutil
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cv2

sid = SentimentIntensityAnalyzer()

dataset = pd.read_csv("dataset.csv")
success = 0

def download(url):
    success = 0
    response = requests.get(url, stream=True)
    with open('img.png', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
        success = 1
    out_file.close()
    del response
    return success
'''
imgs = dataset['urlImage']
user = dataset['username']
desc = dataset['descriptionProfile']
count = 0
for i in range(len(desc)):
    if imgs[i] is not None:
        img_name = imgs[i]
        if ',' not in img_name:
            msg = str(desc[i])
            sentiment_dict = sid.polarity_scores(msg)
            compound = sentiment_dict['compound']
            success = download(img_name)
            try:
                if success == 1:
                    img = cv2.imread("img.png")
                    if compound >= 0.05:
                        cv2.imwrite("images/influence/"+str(count)+".png", img)
                    else:
                        cv2.imwrite("images/non-influence/"+str(count)+".png", img)
                    count = count + 1
            except Exception:
                pass        
    print(str(i)+" "+str(count)+" "+imgs[i])        
'''    
count = 0    
tweets = dataset['tweets']
imgs = dataset['image']
ty = 0
for i in range(len(imgs)):
    if imgs[i] != 'none':
        img_name = imgs[i]
        if ',' not in img_name:
            msg = str(tweets[i])
            sentiment_dict = sid.polarity_scores(msg)
            compound = sentiment_dict['compound']
            success = download(img_name)
            try:
                if success == 1:
                    img = cv2.imread("img.png")
                    if compound >= 0.05:
                        ty = 0
                        cv2.imwrite("images/influence/"+str(count)+".png", img)
                    else:
                        ty = 1
                        cv2.imwrite("images/non-influence/"+str(count)+".png", img)
                    count = count + 1
            except Exception:
                pass        
    print(str(i)+" "+str(count)+" "+imgs[i]+" "+str(ty))
