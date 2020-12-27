import sys
import re
import os
from nltk.stem import SnowballStemmer
from time import perf_counter
import heapq
import math
import bisect
stemmer = SnowballStemmer("english")
path = sys.argv[1]
index_path = "/home/pradeep/Desktop/IRE/wiki/inverted_index/"

# static
sec= {} #secondary index container
sec_page = {} # secondary index for pages
N=0   # no. of processed docs

#dynamic
curr_page = None
vocab = {}
pageids = {} # maintain dictionary of pageids

def _id(num):
    global sec_page
    global pageids
    if num not in pageids:
        l = binarySearch(list(sec_page.keys()), 0, len(sec_page.keys())-1, num)
        f = open(sec_page[l],"r")
        pageids ={}
        for i in f:
            wl= i.rstrip("\n").split(":")
            word = wl[0]
            _list = ":".join(wl[1:])
            pageids[int(word)] = _list
        f.close()
    return pageids[num]
def pretty_print(docs, t, k):
    for i in docs:
        print(str(i[1])+", "+_id(i[1]))
    print(t,t/int(k))

def binarySearch(arr, l, r, x): 
    while l < r: 
        mid = r - (r - l) // 2
        if arr[mid] < x: 
            l = mid
        else: 
            r = mid - 1
    return arr[l]
    # pos = bisect.bisect(arr,x,l,r)
    # return arr[pos-1]
def fetch_sec_index():
    global sec
    if(os.path.exists(index_path+"sec_index.txt")):
        file= open(index_path+"sec_index.txt","r")
    else:
        print("Error: secondary index not built")

    for i in file:
        word, f = i.split(":")
        sec[word.strip()] = f.rstrip("\n")
    file.close()

    if(os.path.exists(index_path+"sec_index_t.txt")):
        file= open(index_path+"sec_index_t.txt","r")
    else:
        print("Error: secondary index for pages not built")
    for i in file:
        word, f = i.split(":")
        sec_page[int(word)] = f.strip()
    file.close()

def fetch_page(page):
    global vocab
    vocab ={}
    filename = sec[page]
    file = open(filename,"r")
    for i in file:
        word, _list= i.split(":")
        vocab[word.strip()] = _list.strip()
    file.close()
    
def get_postinglist(word):
    word = stemmer.stem(word.strip().lower())
    global curr_page
    global vocab
    page = binarySearch(list(sec.keys()), 0, len(sec)-1, word)
    if(page != curr_page):
        fetch_page(page)
        curr_page = page
    if(word in vocab):
        return vocab[word]
    return None

def give_weights(a,weights,f):
    if(a == None):
        return weights
    global N
    dem= {'t':0,'i':1,'b':2,'l':3,'r':4,'c':5}
    df=[0,0,0,0,0,0]
    wd=[4,4,2,1,1,3]
    if(f != None):
        wd[dem[f]] = 10*wd[dem[f]]
    for i in dem.keys():
        df[dem[i]] = a.count(i)

    b = a.split("d")
    for i in b:
        if(i==""):
            continue
        ind = list(filter(None, re.split("[^a-z]",i)))
        v = list(filter(None, re.split("[a-z]",i)))
        w = 0
        for j in range(len(ind)):
            field = dem[ind[j]]
            tf = 1+math.log10(int(v[j+1]))
            idf = 0
            if(df[field] != 0):
                idf = math.log10(N/df[field])
            w+= wd[field]*tf*idf
        if(int(v[0]) not in weights):
            weights[int(v[0])] = w
        else:
            weights[int(v[0])] *= 10*w
    return weights

def normal_query(query):
    a = list(filter(None, query.split(" ")))
    weights = {}
    for i in a:
        _list = get_postinglist(i)
        if(_list !=  None):
            weights = give_weights(_list,weights,None)
    return weights

def field_query(query):
    weights = {}
    s = query.strip().split(" ")
    field = None
    for i in s:
        if(i.find(":") != -1):
            q = i.split(":")
            field = q[0]
            _list = get_postinglist(q[1])
            weights = give_weights(_list,weights,q[0])
        else:
            _list = get_postinglist(i)
            weights = give_weights(_list,weights,field)
    return weights

def getk(docs,k):
    h = []
    ret = []
    for i in docs:
        heapq.heappush(h,[-1*docs[i],i])
    for i in range(int(k)):
        if(len(h)):
            ret.append(heapq.heappop(h))
    return ret

####### main ##########
fetch_sec_index()
file = open(index_path+"N.txt")
N = int(file.readline().strip())
file.close()

file = open(path,"r")
for i in file:
    tstart = perf_counter()
    k, query = i.split(",")
    docs = None
    if(query.find(":") != -1):
        docs = field_query(query)
    else:
        docs = normal_query(query)
    topk = getk(docs,k) 
    telapsed = perf_counter() - tstart
    pretty_print(topk,telapsed,k)
file.close()
