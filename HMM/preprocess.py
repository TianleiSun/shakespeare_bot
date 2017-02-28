import nltk
from nltk.corpus import cmudict
d = cmudict.dict()

def numVowel(li):
    syl = ""
    for elem in li:
        if "1" in elem or "2" in elem: syl += "1"
        elif "0" in elem: syl += "0"
    return syl

def getRhyme(li):
    rhy = ""
    v = 0
    for i in range(len(li) - 1, -1, -1):
        if "1" in li[i] or "2" in li[i] or "0" in li[i]: 
            v = i
            break
    for j in range(v, len(li)):
        rhy += li[j]
    return rhy    

def preprocess_dic(filename):
    word_dic = {} # (word, [tag, [syllable], [rhyme], (son_num, line_num, word_num)])
    syl_dic = {} # (syllable, [words])
    rhy_dic = {} # (rhyme, [words])
    freq_dic = {} # (word, freq)
    
    line_num = 1
    son_num = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.lower()
            line = line.strip()
            try:   
                int(line) # a new sonnet
                line_num = 1 
                son_num += 1
                continue
            except ValueError:
                pass       
            if len(line) > 0:
                tokens = nltk.word_tokenize(line)
                for token in tokens:
                    if token == "'s":
                        tokens.remove(token)
                tagged = nltk.pos_tag(tokens)
                L = len(tagged)
                for i in range(L):
                    word = tagged[i][0]
                    if word not in word_dic: 
                        tag = tagged[i][1]
                        syl = []
                        rhyme = []
                        freq_dic[word] = 1
                        if word in d:
                            for pron in d[word]:
                                syl.append(numVowel(pron)) # list of string ["10", "010"]
                                rhyme.append(getRhyme(pron))  # list of string ['AH1M', 'ER0M']
                            word_dic[word] = [tag, syl, rhyme]
                            for s in syl:
                                if s not in syl_dic:
                                    syl_dic[s] = []
                                if word not in syl_dic[s]:
                                    syl_dic[s].append(word)
                            for r in rhyme:
                                if r not in rhy_dic:
                                    rhy_dic[r] = []
                                if word not in rhy_dic[r]:
                                    rhy_dic[r].append(word)
                    else: 
                        freq_dic[word] += 1
            line_num += 1
        return word_dic, syl_dic, rhy_dic, freq_dic

def preprocess_word_to_num(filename):
    last_word = []
    last_word_seq = []
    last_word_map = {}
    last_word_counter = 0

    word_seq = []
    words = []
    word_map = {}
    word_counter = 0

    line_num = 0
    son_num = -1
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            line = line.lower()
            line = line.strip()
            try:   
                int(line) # a new sonnet
                line_num = 0 
                son_num += 1
                continue
            except ValueError:
                pass       
            if len(line) > 0:
                tokens = nltk.word_tokenize(line)
                #print tokens
                #print "******************************"
                word_seq = []
                for word in tokens:
                    if (word != ')') and (word != "'s") and (word != ',') and (word != ':') and (word != '.') and (word != '?') and (word != '!') and (word != ';'):
                # Add new moods to the mood state hash map.
                        if word not in word_map:
                            word_map[word] = word_counter
                            word_counter += 1
                        word_seq.append(word_map[word])
                word_seq.reverse()

                if(son_num != 98) and (son_num != 125):
                    if word_seq[0] not in last_word_map:
                        last_word_map[word_seq[0]] = last_word_counter
                        last_word_counter += 1
                    last_word_seq.append(last_word_map[word_seq[0]])
                if(line_num == 13) and (len(last_word_seq) > 0):
                    last_word.append(last_word_seq)
                    last_word_seq = []
                words.append(word_seq) 
                line_num = line_num + 1


    return words, last_word, word_map, last_word_map