import csv
import nltk
from matplotlib import pyplot as plt
from transformers import pipeline
from datetime import datetime
import numpy as np

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

class Review(object):
    def __init__(self, ID, asin, username, helpful, text, rating, summary, unixtime, date, llm):
        self.ID = ID
        self.asin = asin
        self.username = username
        self.helpful = helpful
        self.text = text
        self.rating = float(rating)
        self.summary = summary
        self.unixTime = int(unixtime)
        self.date = datetime.utcfromtimestamp(int(unixtime))
        self.n = 0 #for counting purposes
        self.priorT = int(unixtime)
        self.dt = []
        if llm == "True":
            self.llm = True
        else:
            self.llm = False



sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."

tokens = nltk.word_tokenize(sentence)
print(tokens)


labels = []
reviews = []

with open('Amazon_reviews_plus_LLM.csv', mode ='r')as file:

  # reading the CSV file
  csvFile = csv.reader(file)

  # displaying the contents of the CSV file
  for lines in csvFile:
    if lines[0] == "reviewerID":
        continue
    reviews.append(Review(*lines))

def analyze_sentiment(review):
    result = sentiment_model(review)[0]
    return result["label"], result["score"]

# Apply sentiment analysis to a few example reviews
example_reviews = [review.text for review in reviews[:30]]
sentiments = [analyze_sentiment(review) for review in example_reviews]

for i, (review, sentiment) in enumerate(zip(example_reviews, sentiments)):
    #print(f"Review {i + 1}:")
    print(f"Text: {review}")
    print(f"Sentiment: {sentiment[0]}, Score: {sentiment[1]:.2f}")
    #print()
    

#%% looking at time
plt.close('all')

# sort reviews by date reviewed
reviews.sort(key = lambda x: x.unixTime, reverse = True)

rreal = [r for r in reviews if not r.llm]
rllm = [r for r in reviews if r.llm]

rIDllm = set([r.ID for r in reviews if r.llm]) #list of IDs that are llm
rID = set([r.ID for r in reviews if not r.llm]) # list of IDs that are real

nreal = len(rID)
nllm = len(rIDllm)

print(rID.intersection(rIDllm)) # check if there are imposters lol


timesreal = np.array([review.unixTime for review in rreal])
timesllm = np.array([review.unixTime for review in rllm])


"""
# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(timesllm, bins = 50)
# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(times, bins = 50)
 
# Show plot
plt.show()"""


# look at average time in between reviews for a given reviewer

rrealtime = dict()
rllmtime = dict()
        
        

counts = dict()
for r in rreal:
    if r.ID in counts.keys():
        counts[r.ID] += 1
    else:
        counts[r.ID] = 1


#histogram for real people
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(counts.values(), bins = 200, density = True)
ax.set_xlim((0,40))
ax.set_xlabel("Total reviews")
ax.set_ylabel("Percentage")
ax.set_title("Real Reviews")


countsllm = dict()
for r in rllm:
    if r.ID in countsllm.keys():
        countsllm[r.ID] += 1
    else:
        countsllm[r.ID] = 1


#histogram for llm people
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(countsllm.values(), bins = 60, density = True)
ax.set_xlim((0,20))
ax.set_xlabel("Total reviews")
ax.set_ylabel("Percentage")
ax.set_title("LLM Reviews")


#%% looking at average dt in between reviews
rreal.sort(key = lambda x: x.unixTime, reverse = True)
rllm.sort(key = lambda x: x.unixTime, reverse = True)


rrealtimes = dict()
rllmtimes = dict()
rrealtimes_save = [rID for rID in counts.keys() if counts[rID] > 1] # list of keys
rllmtimes_save = [rID for rID in countsllm.keys() if countsllm[rID] > 1] # list of keys

for r in rreal:
    if r.ID in rrealtimes_save: # if this reviewer id has multiple reviews
        if r.ID in rrealtimes.keys():
            #deltat = rllmtimes[r.ID][-1] - r.unixTime
            rrealtimes[r.ID].append(r.unixTime)
            #print('hi')
        else:
            rrealtimes[r.ID] = [r.unixTime]
            #print('ok')
for r in rllm:
    if r.ID in rllmtimes_save: # if this reviewer id has multiple reviews
        if r.ID in rllmtimes.keys():
            #deltat = rllmtimes[r.ID][-1] - r.unixTime
            rllmtimes[r.ID].append(r.unixTime)
            #print('hi')
        else:
            rllmtimes[r.ID] = [r.unixTime]
            #print('ok')
        
#%%
plt.close('all')


dtreal = dict()
for k,v in rrealtimes.items():
    for i in range(len(rrealtimes[k])-1):
        dt = -rrealtimes[k][i+1] + rrealtimes[k][i]
        if k in dtreal.keys():
            dtreal[k].append(dt)
        else:
            dtreal[k] = [dt]

dtllm = dict()
for k,v in rllmtimes.items():
    for i in range(len(rllmtimes[k])-1):
        dt = -rllmtimes[k][i+1] + rllmtimes[k][i]
        if k in dtllm.keys():
            dtllm[k].append(dt)
        else:
            dtllm[k] = [dt]
    
for k,v in dtreal.items():
    dtreal[k] = np.std(dtreal[k])
    
for k,v in dtllm.items():
    dtllm[k] = np.std(dtllm[k])
    
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(dtllm.values(), bins = 500)
#ax.set_xlim((0,1e8))
ax.set_xlabel("standard deviation dt [seconds]")
ax.set_ylabel("Number of Users")
ax.set_title("LLM Reviews")

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(dtreal.values(), bins = 500)
#ax.set_xlim((0,5e7))
ax.set_xlabel("standard deviation dt [seconds]")
ax.set_ylabel("Number of Users")
ax.set_title("real Reviews")
