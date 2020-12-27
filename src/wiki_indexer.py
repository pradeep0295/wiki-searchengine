import xml.sax
import sys
import os
import re
import heapq
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize.regexp import regexp_tokenize

## all paths
dump = sys.argv[1]
inx_loc = sys.argv[2]
if not os.path.exists(inx_loc):
    os.mkdir(inx_loc)
if not os.path.exists(inx_loc+"un_merged/"):
    os.mkdir(inx_loc+"un_merged/")
stat = sys.argv[3]

## Cache
cachedStopWords = stopwords.words("english")
stemmer = SnowballStemmer("english")
hash_stem = {}

## global variables
gdic = {}
g_pos = []
count = 0
id_ti = []
cf = 0
merge = {}
written = 0
limit = 50*1024*1024
splits = 0

## Compress Indexing
delimiters = {0:'t',1:'i',2:'b',3:'l',4:'r',5:'c'}
def compress(_dict):
    s = ""
    for i in _dict:
        s = s+"d"+str(i)
        for j in range(6):
            if(_dict[i][j] > 0):
                dlim = delimiters[j]
                s= s+dlim+str(_dict[i][j])
    return s

## helper utils
def pattern(a, po, pc, field):
    found = 0
    bal = 0
    category = ""
    i=0
    for i in range(len(a)):
        if(found!=0 and a[i] == po):
            bal+=1
        elif(found !=0 and a[i] == pc):
            bal-=1
        if(found !=0 and bal == 0):
            category += a[found:i]
            found = 0
            if(field == 'infobox'):
                break
        if(a[i] == po and i+1<len(a) and a[i+1] == po and i+2+len(field)<len(a) and a[i+2:i+2+len(field)] == field):
            found = i+len(field)+3
            bal+=1
    return category,i

def cite(test):
    a = ""
    found =0
    for i in range(len(test)):
        if(found == 0 and test[i] == '='):
            found = i+1
        if(found!=0 and (test[i]=='|' or test[i] == '}')):
            t = test[found:i]
            if(t[:4] !="http"):
                a+=t
            found=0
    return a

def Tokenize(s):
    global hash_stem
    words = regexp_tokenize(s,pattern="[a-z]+ | [0-9]+")
    ret = []
    for word in words:
        if(len(word)>2 and len(word)<41 and word not in cachedStopWords):
            word = word.strip()
            if word not in hash_stem:
                hash_stem[word] = stemmer.stem(word)   
            ret.append(hash_stem[word])
            # if(len(hash_stem) == 100000000):
            #     print(len(hash_stem))
            #     hash_stem = {}
    return ret

## final tokens with resp. fields
def wordlist(page):
    ibox, skip = pattern(page[0][1],'{','}',"infobox")
    text = page[0][1]
    
    b = page[0][1].find("==external links")
    a = page[0][1].find("==references")
    c = page[0][1].find("[[category")
    
    cat = pattern(text[c:],'[',']',"category")[0]

    if(a>0 and b>0 and c>0):
        body, ref, ext, cat = text[skip:a], text[a+16:b], text[b+12:c], cat
    elif(a>0 and b>0 and c<0):
        body, ref, ext, cat = text[skip:a], text[a+16:b], text[b+12:c],""
    elif(a>0 and b<0):
        body, ref, ext, cat = text[skip:a], text[a+16:b], "", ""
    else:
        body, ref, ext, cat = text[skip:],"","",""
    ibox = Tokenize(ibox)
    title = Tokenize(page[0][0])
    body = Tokenize(body)
    ext = Tokenize(re.sub("\[.*\]"," ",ext))
    ref = Tokenize(" ".join(re.findall("\*[^=]*\n",ref))+cite(ref))
    cat = Tokenize(cat)
    return title,ibox,body,ext,ref,cat

## Inverted Index Creation
def Update_Index(page,docid):
    global gdic
    global g_pos
    dictionary = {}
    pos_list = []
    article = wordlist(page)
    for field in range(len(article)):
        global count
        count+=len(article[field])
        for i in range(len(article[field])):
            if article[field][i] not in dictionary:
                dictionary[article[field][i]] = [1,len(pos_list)]
                temp = [0,0,0,0,0,0]
                temp[field]+=1
                pos_list.append({docid:temp})
            else:
                dictionary[article[field][i]][0] +=1
                index = dictionary[article[field][i]][1]
                if docid not in pos_list[index].keys():
                    temp = [0,0,0,0,0,0]
                    temp[field]+=1
                    pos_list[index].update({docid:temp})
                else:
                    pos_list[index][docid][field] +=1
    for i in dictionary:
        if i not in gdic:
            gdic[i] = [1,len(g_pos)]
            g_pos.append(compress(pos_list[dictionary[i][1]]))
        else:
            gdic[i][0]+= 1
            g_pos[gdic[i][1]]+= compress(pos_list[dictionary[i][1]])
            
class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []
        self._npages = 0

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        global id_ti
        """Closing tag of element"""
        if name == self._current_tag:
            if(len(self._values) == 2):
                self._values = {}
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._pages = []
            self._pages.append((self._values['title'].lower(), self._values['text'].lower())) ##immediate lower
            id_ti.append(str(self._npages)+":"+self._pages[0][0]+"\n")
            Update_Index(self._pages,self._npages) #Here
            self._npages += 1

def write_to_disk(c):
    file = open(inx_loc+"un_merged/"+"index"+str(c)+".txt","w+")
    for i in sorted(gdic.keys()):
        file.write(i+":")
        file.write(g_pos[gdic[i][1]])
        file.write("\n")
    file.close()
    idf = open(inx_loc+"id_title"+str(c)+".txt","w+")
    for i in id_ti:
        idf.write(i)
    idf.close()

def write_and_split(word,_list):
    global f
    global splits
    global limit
    global written
    written += f.write(word.rstrip()+":"+_list.lstrip()+"\n")
    if(written > limit):
        written = 0
        f.close()
        splits+=1
        f = open(inx_loc+str(splits)+".txt","w+")

########### main  #############
handler = WikiXmlHandler()
parser = xml.sax.make_parser()
parser.setContentHandler(handler)
# parser.parse(dump)
# write_to_disk(0)
size = 0 
for i in os.listdir(dump):
    print(i)
    parser.parse(dump+i)
    write_to_disk(cf)
    size+=len(gdic)
    gdic = {}
    g_pos = []
    id_ti = []
    cf+=1
    hash_stem = {}

### Merge Sort files and split
fp =[]  ## file pointers
h = []  ## heap
f = open(inx_loc+str(splits)+".txt","w+")
for i in range(cf):
    fp.append(open(inx_loc+"un_merged/"+"index"+str(i)+".txt","r"))
for i in range(len(fp)):
    w,l = fp[i].readline().rstrip("\n").split(":")
    h.append((w,i,l))
    heapq.heapify(h)
while(len(h)):
    w, i, l = heapq.heappop(h)
    if(len(h)>0 and w == h[0][0]):
        tw,ti,tl = heapq.heappop(h)
        heapq.heappush(h,(tw,ti,tl+l))
    else:
        write_and_split(w,l)
    if(fp[i].closed):
        te = ""
    else:
        te = fp[i].readline()
    if(te != ""):
        tw,tl = te.rstrip("\n").split(":")
        heapq.heappush(h,(tw,i,tl))
    else:
        fp[i].close()

####close all files
f.close()

aan = open(inx_loc+"N.txt","w+")
aan.write(str(handler._npages))
aan.close()

s_index = open(inx_loc+"sec_index.txt","w+")
for i in range(0,splits+1):
    f = open(inx_loc+str(i)+".txt","r")
    start = f.readline().split(":")[0]
    s_index.write(start+":"+inx_loc+str(i)+".txt"+"\n")
    f.close()
s_index.close()

s_index = open(inx_loc+"sec_index_t.txt","w+")
for i in range(0,cf):
    f = open(inx_loc+"id_title"+str(i)+".txt","r")
    start = f.readline().split(":")[0]
    s_index.write(start+":"+inx_loc+"id_title"+str(i)+".txt"+"\n")
    f.close()
s_index.close()

file = open(stat,"w+")
file.write(str((splits+1)*limit)+"\n")
file.write(str(splits+1)+"\n")
file.write(str(size))
file.close()
