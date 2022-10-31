def read_sents(fname='train.zh-en'):
    '''Read the lines from the file.
    Each line is Chinese\tEnglish.
    Output a list of Chinese sentences and a list of English sentences.
    Every sentence in the list should be given as a list of tokens.
    English sentences should all start with 'NULL'
    '''
    data = []
    ch_sents = []
    en_sents = []
    for line in open(fname,'r'):
        splits = line.rstrip().split('\t',2)
        ch_sents.append(splits[0].split(' '))
        en_sents.append(list(['NULL'] + splits[1].split(' ')))
        data.append((splits[0].strip().split(' '),splits[1].strip().split(' ')))
    return ch_sents, en_sents, data

ch_sents, en_sents, data = read_sents()

def translation_model(ch_sents, en_sents):
    '''Given sentences in the format above, initialize a word-for-word translation model.
    tfe '''
    tfe = {(v1,v2):0.00001  for val1, val2 in zip(ch_sents,en_sents) for v1 in val1 for v2 in val2}
    return tfe

tfe = translation_model(ch_sents, en_sents)
'''Format of tfe: translation weight of Chinese token f given English token e
tfe[Chinese token, English token] = translation weight
tfe['船长', 'NULL'] = 0.5
'''

def estep(en_sents, ch_sents, tfe):
    '''Given the training sentences in the form above and our current translation model, perform the E step.
    '''
         
    cfe = {}    
    
    for ch_sent, en_sent in zip(ch_sents, en_sents):      
        prior = {}
                
        for ch in ch_sent:
            for en in en_sent:
                if ch in prior:
                    prior[ch] += tfe[(ch,en)]
                else:
                    prior[ch] = tfe[(ch,en)]
        
        for ch in ch_sent:
            for en in en_sent: 
                if (ch,en) in cfe:
                    cfe[(ch,en)] += (tfe[(ch, en)] / prior[ch])
                else:
                    cfe[(ch,en)] = (tfe[(ch, en)] / prior[ch])
                
    return cfe


def mstep(cfe):
    '''Given the output of the E step, perform the M step.'''

    tfe = {}
    tot = {}
    
    for (ch, en) in cfe:
        if en in tot:
            tot[en] += cfe[(ch,en)]
        else:
            tot[en] = cfe[(ch,en)]
    
    for (ch, en) in cfe:
        tfe[(ch, en)] = cfe[(ch, en)] / tot[en]    
    
    return tfe


import math
def likelihood(tfe, en_sents, ch_sents):
    '''Compute the log-likelihood of the data'''
    prob = 0
    for ch, en in zip(ch_sents, en_sents):
        product = 1
        for cw in ch:
            sm = 0
            for ew in en:
                sm += tfe[(cw, ew)]
            product *= 1/(len(en)) * sm
        P = 1/100 * product
        prob += math.log(P)
    return prob


def train(en_sents, ch_sents, tfe, steps=10):
    '''Train by repeating the E and M steps'''
    for i in range(steps):
        print(f'Epoch ::{i}')
        estep = estep(en_sents, ch_sents, tfe)
        mstep = msptep(estep)
        tfe = mstep        
        print(likelihood(tfe, en_sents, ch_sents))
    return tfe


def write_ttable(tfe, fname='tfe-dump.out'):
    '''Given our trained word-for-word translation table,
    dump the contents to a file in the format:
    English Chinese translation_weight
    ex.
    bring 只不过 0.0012009567030861706'''
    with open(fname, "a") as file_out:
        for k in tfe:
            file_out.write(f'{k[1]} {k[0]} {tfe[k]} \n')