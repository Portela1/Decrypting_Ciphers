import numpy as np
import matplotlib.pyplot as plt

import string
import random
import re
import requests
import os
import textwrap


###
# Mapping
###

#Creating both dictionarys for the cipher
dictionary1 = list(string.ascii_lowercase)
dictionary2 = list(string.ascii_lowercase)

#Creating the dictionary where the mapping will go
real_mappping = {}
#Creating the dictionary for decoding messeges
decode_mapping ={}

#Randomizing one dictionary to make it exiting
random.shuffle(dictionary2)


#Creat the mapping
for k, v in zip(dictionary1, dictionary2):
    real_mappping[k] = v
for k, v in zip(dictionary2, dictionary1):
    decode_mapping[k] = v


### 
# Language model
###

# Initialize Markov Matrix
Matrix = np.ones((26,26))

# Inistialize stae distribution
Pi = np.zeros(26)

# Function to update Markov Matrix
def update_markov(ch1,ch2):
    # ord('a') = 97 // ord('d') = 100
    i = ord(ch1) - 97
    j = ord(ch2) - 97

    Matrix[i,j] += 1

# Funciton to update Pi
def update_pi(ch):
    i = ord(ch) - 97
    Pi[i] += 1


#Calculate probability of word
def calculate_prob_word(word):
    i = ord(word[0]) - 97
    logP = np.log(Pi[i])

    for ch in word[1:]:
        j = ord(ch) - 97
        logP += np.log(Matrix[i,j])
        i = j 

    return logP

#Calculate probability of a sequence sequence of words
def calculate_prob_sequence(words):

    #Converd sentence in a string to a list of string words (to accept two inputs)
    if type(words) == str:
        words = words.split()

    logP = 0

    #Calculate the probabillity of a sequence of words
    for word in words:
        logP += calculate_prob_word(word)
    return logP



###
# Creating a markof model based on the English dataset
###

# Download file
if not os.path.exists('moby_dick.txt'):
    print("Downloading moby dick...")
    r = requests.get("https://lazyprogrammer.me/course_files/moby_dick.txt")
    with open('moby_dick.txt','w') as f:
        f.write(r.content.decode())

# Finding non aplpa characters
regex = re.compile('[^a-zA-Z]')

# Load in words
for line in open('moby_dick.txt'):
    line = line.rstrip()


    # Only go to lines with actual words in them
    if line:
        line = regex.sub(' ', line) #replace non alpha characters with ' ' space


        # lower case everithing on the line
        tokens = line.lower().split()

        for token in tokens:
            # Update Matrix and Pi

            # First letter
            ch0 = token[0]
            update_pi(ch0)

            for ch1 in token[1:]:
                update_markov(ch0,ch1)
                ch0 = ch1

    # Normalize the probabilities
Pi /= Pi.sum()
Matrix /= Matrix.sum(axis = 1, keepdims = True)



###
# Massage to use
###

message_to_use = '''Stand not by me, but stand under me, whoever you are that will now
help Stubb; for Stubb, too, sticks here. I grin at thee, thou grinning
whale! Who ever helped Stubb, or kept Stubb awake, but Stubb’s own
unwinking eye? And now poor Stubb goes to bed upon a mattrass that is
all too soft; would it were stuffed with brushwood! I grin at thee,
thou grinning whale! Look ye, sun, moon, and stars! I call ye assassins
of as good a fellow as ever spouted up his ghost. For all that, I would
yet ring glasses with ye, would ye but hand the cup! Oh, oh! oh, oh!
thou grinning whale, but there’ll be plenty of gulping soon! Why fly ye
not, O Ahab! For me, off shoes and jacket to it; let Stubb die in his
drawers! A most mouldy and over salted death, though;—cherries!
cherries! cherries! Oh, Flask, for one red cherry ere we die!'''

###
# Massange encoding
###

def encode(msg):
    msg = msg.lower()

    msg = regex.sub(' ', msg)

    coded_msg = []
    for ch in msg:
        coded_ch = ch
        if ch in real_mappping:
            coded_ch = real_mappping[ch]
        coded_msg.append(coded_ch)

    return ''.join(coded_msg)

# Encoded Message
encoded_message = encode(message_to_use)


###
# Message decoding
###
def decode(msg,word_map):
    decoded_msg = []
    for ch in msg:
        decoded = ch
        if ch in word_map:
            decoded = word_map[ch]
        decoded_msg.append(decoded)

    return ''.join(decoded_msg)


###
# Evolutionary algorithm
###

dna_pool = []

for _ in range(20):
    dna = list(string.ascii_lowercase)
    random.shuffle(dna)
    dna_pool.append(dna)
    


def procreate(dna_pool,n_children):

    offspring = []
    for dna in dna_pool:
        for _ in range(n_children):
            copy = dna.copy()
            j = np.random.randint(len(copy))
            k = np.random.randint(len(copy))


            tmp = copy[j]
            copy[j] = copy[k]
            copy[k] = tmp
            offspring.append(copy)

    return offspring + dna_pool



num_iters = 1000
scores = np.zeros(num_iters)
best_dna = None
best_map = None
best_score = float('-inf')

for i in range (num_iters):

    if i > 0:
        dna_pool = procreate(dna_pool,3)


    dna2score = {}
    for dna in dna_pool:
        current_map = {}
        for k,v in zip(dictionary1,dna):
            current_map[k] = v

        decoded_messege = decode(encoded_message,current_map)
        score = calculate_prob_sequence(decoded_messege)

        dna2score[''.join(dna)] = score

        if score > best_score:
            best_dna = dna
            best_map = current_map
            best_score = score
    
    scores[i] = np.mean(list(dna2score.values()))



    sorted_dna = sorted(dna2score.items(), key = lambda x:x[1], reverse = True)
    dna_pool = [list(k) for k, v in sorted_dna[:5]]

    if i % 200 == 0:
        print("iter:", i, " score:", scores[i], "best so far", best_score)

decoded_messege = decode(encoded_message,best_map)

print("LL of decode message:", calculate_prob_sequence(decoded_messege))
print("LL of true message:" , calculate_prob_sequence(decode(encoded_message,decode_mapping)))

for true,v in real_mappping.items():
    pred = best_map[v]
    if true != pred:
        print("true: %s, pred: %s" % (true,pred))




print("####################  Encoded messege  ########################")
print(encoded_message)
print("###############################################################")
print("###############################################################")
print("###############################################################")
print("###############################################################")
print(" ")
print(decode(encoded_message,best_map))


plt.plot(scores)
plt.show()