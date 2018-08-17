# -*-coding:utf-8-*-
'''
Created on 2018骞�8鏈�4鏃� 涓嬪崍7:59:37

@author: liusheng1
'''

def extract_docs(texts, isTrain = True):
    header = []
    docs = []
    y = []
    with open(texts, encoding = 'utf-8') as f:
        for ind, line in enumerate(f):
            if ind == 0:
                header = line.strip().split(',')
                continue
            line_tmp = line.strip().split(',')
            docs.append(line_tmp[2])
            if isTrain:
                y.append(line_tmp[-1])
    if isTrain:
        return docs, y
    else:
        return docs
    
def extract_word_in_docs(docs):
    words = []
    for doc in docs:
        wordLst = doc.split(' ')
        words.extend(wordLst)
        
    return words

def print_longest_lowest(docs):
    longest = 0
    lowest = 99999
    for ind, doc in enumerate(docs):
        doc_len = len(doc.split(' '))
        if longest < doc_len:
            longest = doc_len
        if lowest > doc_len:
            lowest = doc_len
    
    print(longest, lowest)
   
def plot_length_dist(docs):
    length_lst = []
    for ind, doc in enumerate(docs):
        doc_len = len(doc.split(' '))
        length_lst.append(doc_len)
    
    length_lst = sorted(length_lst)
#     pdb.set_trace()
    plt.bar(range(len(length_lst)), length_lst, color='rgb')
    plt.show()

def analyseLength(train_path, test_path):
    train_docs, _ = extract_docs(train_path)
    
    print_longest_lowest(train_docs)
    plot_length_dist(train_docs)
    del train_docs
    
    test_docs = extract_docs(test_path, isTrain = False)
    print_longest_lowest(test_docs)
    plot_length_dist(test_docs)

def uniqWords(docs, length):
    doc_Words = []
    for ind, doc in enumerate(docs):
        splitted_doc = doc.split(' ')
        if len(splitted_doc) <= length:
#             pdb.set_trace()
            doc_Words.extend(splitted_doc)
    doc_uniqWords = set(doc_Words)
    return doc_uniqWords

def getUniqWord(train_path, test_path, length = 30):
    train_docs, _ = extract_docs(train_path)

    train_uniqWords = uniqWords(train_docs, length)
    print("train docs which length was smaller than %d has %d uniq words in total." 
          % (length, len(train_uniqWords)))

    del train_docs
    test_docs = extract_docs(test_path, isTrain = False)
    test_uniqWords = uniqWords(test_docs, length)
    print("test docs which length was smaller than %d has %d uniq words in total." 
          % (length, len(test_uniqWords)))

def extractShortTextUniqWords(docs, length):
    uniq_words = []
    for doc in docs:
        splitted_doc = doc.split(' ')
        if len(splitted_doc) <= length:
            uniq_words.extend(splitted_doc)
    return set(uniq_words)

def keepUniqWordsOfShortTexts(train_path, test_path, length = 30):
    train_docs, _ = extract_docs(train_path)
    train_uniqWords = extractShortTextUniqWords(train_docs, length); del train_docs
    test_docs = extract_docs(test_path, isTrain = False)
    test_uniqWords = extractShortTextUniqWords(test_docs, length)
    
#     print(",".join(w for w in train_uniqWords.union(test_uniqWords)))
    return train_uniqWords.union(test_uniqWords)

def extractWords(docs):
    uniq_words = []
    for doc in docs:
        splitted_doc = doc.split(' ')
        uniq_words.extend(splitted_doc)
        
    return set(uniq_words)


def setVoc(train_class_set, test_words_set):
    return sorted(train_class_set.union(test_words_set))

def isTestHasDifferentWordsInTrain(train_words_set, test_words_set):
    cnt = 0
    for word in test_words_set:
        if word not in train_words_set:
            cnt += 1
#             print("test_set has %d words and word %s is not in train_set" % (cnt, word))
    print("test_set has %d words not in train_set" % cnt)

