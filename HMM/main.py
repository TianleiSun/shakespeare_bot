#################################################################
##           Functions
#################################################################       
def generate_emission(A, O):
    emission = []
    state = random.choice(range(hidden_states))

    for t in range(7):
        # Sample next observation.
        new_word = ""
        while (True):
            rand_var = random.uniform(0, 1)
            next_obs = 0

            while rand_var > 0:
                rand_var -= O[state][next_obs]
                next_obs += 1

            next_obs -= 1
            word_id = last_map.keys()[last_map.values().index(next_obs)]
            new_word = word_map.keys()[word_map.values().index(word_id)]
            if new_word in word_dic:
                break
            
        emission.append(new_word)

        # Sample next state.
        rand_var = random.uniform(0, 1)
        next_state = 0

        while rand_var > 0:
            rand_var -= A[state][next_state]
            next_state += 1

        next_state -= 1
        state = next_state

    return emission
    
def generate_line(last_word, A, O):
    syl_seq = "0101010101" 
    syl_list = word_dic[last_word][1]
    num_syl = len(syl_list[0])
    for elem in syl_list:
        if elem == syl_seq[-num_syl:]:
            syl_seq = syl_seq[:len(syl_seq) - num_syl]
            break
            
    
    emission = [last_word]
    state = random.choice(range(hidden_states))
    new_word = ""
    while(len(syl_seq) != 0):
        # Sample next observation.
        flag = False
        while(not flag):
            wlist = []
            for i in range(100):
                rand_var = random.uniform(0, 1)
                next_obs = 0

                while rand_var > 0:
                    rand_var -= O[state][next_obs]
                    next_obs += 1

                next_obs -= 1
                new_word = word_map.keys()[word_map.values().index(next_obs)]
                wlist.append(new_word)
            s = set(wlist)
            new_word = random.sample(s, 1)[0]

            
            flag = False
            if new_word not in word_dic: continue
            syl_list = word_dic[new_word][1]
            num_syl = len(syl_list[0])
            for elem in syl_list:
                if elem == syl_seq[-num_syl:]:
                    syl_seq = syl_seq[:len(syl_seq) - num_syl]
                    flag = True
                    break
        # print(new_word)
        emission = [new_word] + emission

        # Sample next state.
        rand_var = random.uniform(0, 1)
        next_state = 0

        while rand_var > 0:
            rand_var -= A[state][next_state]
            next_state += 1

        next_state -= 1
        state = next_state

    return emission
    
def generate_lastwords(list_words):
    last_words = []
    gen_words = []
    for word in list_words:
        rhy = word_dic[word][2][0]
        cand_list = rhy_dic[rhy]
        id = random.randint(0, len(cand_list) - 1)
        new_word = cand_list[id]
        while (new_word == word):
            id = random.randint(0, len(cand_list) - 1)
            new_word = cand_list[id]
        gen_words.append(cand_list[id])
    for i in range(0, 6, 2):
        last_words.append(list_words[i])
        last_words.append(list_words[i + 1])
        last_words.append(gen_words[i])
        last_words.append(gen_words[i + 1])
    last_words.append(list_words[6])
    last_words.append(gen_words[6])
    return last_words
    
def generate_sonnet(lastlist, A, O):
    symb = [".","!",".",".", "?", ";", ".", ".", ".",  ".", ".", ".", ".","!",".",".", ".", ";", ".", ".", ".",  ".", ".", "."]
    res = ""
    li = []
    tabrand = random.uniform(0,1)
    for i in lastlist:
        li.append(generate_line(i, A, O))
    for i in range(len(li)):
        if tabrand > 0.7 and i >= 12:
            res += "\t"
        for j in range(len(li[i])):
            if j == len(li[i]) - 1:
                res += li[i][j]
                if i % 2 == 0:
                    res += ","
                else:
                    res += symb[random.randint(0,len(symb)-1)]
                res += "\n"
            else:
                res += (li[i][j] + ' ') 
    return res
############################### Main ###########################
import preprocess
import random
from unsupervise import unsupervised_learning

[word_dic, syl_dic, rhy_dic, freq_dic] = preprocess.preprocess_dic("shakespeare.txt")
words, lastwords, word_map, last_map = preprocess.preprocess_word_to_num("shakespeare.txt")
sequence = lastwords
hidden_states = 10
iterations = 20
lastA, lastO = unsupervised_learning(sequence,hidden_states, iterations)
sequence = words
hidden_states = 10
iterations = 20
wordA, wordO = unsupervised_learning(sequence,hidden_states, iterations)

lastlist = generate_emission(lastA,lastO)
last14 = generate_lastwords(lastlist)
print generate_sonnet(last14, wordA, wordO)


############################### view hidden states ###########################
def get_hidden_states(O):
    hidden_states_word = []
    for j in range (len(O)):
        word_list = []
        temp = O[j]
        m = 30
        dic = {}
        for k in range(len(temp)):
            if len(dic) < m:
                dic[k] = temp[k]
            else:
                (key, minval) = min(dic.items(), key = lambda x: x[1])
                if temp[k] > minval:
                    del dic[key]
                    dic[k] = temp[k]
        for i in range(20, len(dic.keys())):
            word = word_map.keys()[word_map.values().index(dic.keys()[i])]
            word_list.append(word)
        hidden_states_word.append(word_list)
    return hidden_states_word
    
  