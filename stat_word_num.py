import numpy as np
import os
import re
import random
import chardet

import jieba    

from jieba import analyse

from collections import Counter
from jieba import analyse




get_keywords_by_tags = analyse.extract_tags
get_keywords_by_textrank = analyse.textrank


### libs lists ###
def strip_comma_or_other(s):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n ，；：。\xa0（）《》、]+'  
    s = re.sub(r, "", s)
    return s

def get_encoding(fn):
    if_ = open (fn, 'rb')
    ft = chardet.detect(if_.read())
    if_.close()
    return ft["encoding"]

def m_(id_text, id_match):
    ret_v=False
    if re.search( id_match,id_text):
        ret_v=True  
    return ret_v

### end libs ###


#fn = "E:/jd/t/nlp_10.txt"
fn = "./train_set_sample.csv"


# print(get_encoding(fn))

if_ = open(fn, errors="ignore")
cnt = [0]

for i in if_:
    if not m_(i, "word_"):
        vec_ = i.split(",")
        word_seq = vec_[-2]
        kw_tags  =  get_keywords_by_tags(word_seq)
        kw_textr =  get_keywords_by_textrank(word_seq)
        kw_cut = jieba.cut(word_seq)
        kw_cut_top = Counter(kw_cut).most_common(40)
        
        ret_val =  "\n". join([
        "------- " + str(cnt[0]) + " -------",  
        fn,
        "",
        "kw_tags:" + str(kw_tags),
#         "kw_textr" + str(kw_textr),
            "\n",
        "kw_cut_top:"  + str(kw_cut_top)]) + "\n"
        cnt[0] = cnt[0] + 1
        
        print(ret_val)

        
#         print(vec_[-2])
#         print (vec_[-1])
#         print (i) 
        
        
        print ("-----")
    
if_.close()

"""

fc = open(fn, get_encoding(fn)).readlines()
print (fc)


"""






