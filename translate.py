import sys
import collections
import itertools
import math
import lm
import heapq

max_fertility = 3
word_reward = 1.
max_zerofert = 2
zerofert_threshold = 1e-2
word_beam_size = 5
beam_size = 30

verbose = 1

class Item(object):
    def __init__(self, coverage, lmstate, pointer):
        self.coverage = coverage
        self.lmstate = lmstate
        self.pointer = pointer
        
    def __hash__(self):
        return hash((self.coverage, self.lmstate))
    
    def __eq__(self, other):
        return (self.coverage, self.lmstate) == (other.coverage, other.lmstate)
    
    def __repr__(self):
        #return "Item({}, {}, {})".format(self.coverage, self.lmstate, self.pointer)
        return "Item({}, {})".format(self.coverage, self.lmstate)

class Bin(object):
    def __init__(self):
        self.items = {}
        
    def add(self, item, score):
        if item not in self.items:
            self.items[item] = score
            if verbose >= 2:
                print('new item:', newitem, s, file=sys.stderr)
            return True
        elif score > self.items[item]:
            del self.items[item] # need to replace key as well as value
            self.items[item] = score
            if verbose >= 2:
                print('better item:', newitem, s, file=sys.stderr)
            return True
        else:
            return False
        
    def prune(self, size):
        items = {}
        for item in heapq.nlargest(size, self.items, key=self.items.get):
            items[item] = self.items[item]
        self.items = items
    
tmfilename, lmfilename, infilename = sys.argv[1:]

# Train language model

lmdata = []
lmorder = 3
for line in open(lmfilename):
    lmdata.append(line.split())
langmodel = lm.Uniform(lmdata)
for i in range(1, lmorder+1):
    langmodel = lm.KneserNey(lmdata, i, langmodel)

# Load translation model

tinv = {}
for line in open(tmfilename):
    e, f, p = line.split()
    p = float(p)
    tinv.setdefault(f, {})[e] = math.log(p)
t = {}
for f in tinv:
    es = []
    for e, p in tinv[f].items():
        if e != 'NULL':
            p += math.log(langmodel.prob((), e))
            es.append((p, e))
    for p, e in heapq.nlargest(word_beam_size, es):
        t.setdefault(e, {})[f] = tinv[f][e]
    t.setdefault('NULL', {})[f] = tinv[f]['NULL']

zerofert = set()
for e in t:
    if langmodel.prob((), e) >= zerofert_threshold:
        zerofert.add(e)

# Main loop
        
for line in open(infilename):
    fwords = line.split()
    if verbose: print('input sentence:', ' '.join(fwords), file=sys.stderr)
    n = len(fwords)

    # form set of candidate English words; this saves time later
    evocab = set()
    for e in t:
        if any(f in t[e] for f in fwords) and e != 'NULL':
            evocab.add(e)
    
    bins = [Bin() for j in range(n+1)]
    goals = Bin()
    
    bins[0].add(Item(0, ('<s>',)*(lmorder-1), None), 0.)

    for n_covered in range(n+1):
        bins[n_covered].prune(beam_size)

        # Extend items in bin with zero-fertility English words
        items = list(bins[n_covered].items)
        for _ in range(max_zerofert):
            newitems = set()
            for trigger in items:
                if verbose >= 2:
                    print('trigger:', trigger, file=sys.stderr)
                for e in zerofert:
                    s = bins[n_covered].items[trigger]
                    s += math.log(langmodel.prob(trigger.lmstate, e))
                    s += word_reward
                    q = (trigger.lmstate + (e,))[-lmorder+1:]
                    newitem = Item(trigger.coverage, q, trigger)
                    if bins[n_covered].add(newitem, s):
                        newitems.add(newitem)
            if len(newitems) > 0:
                bins[n_covered].prune(beam_size)
                items = newitems.intersection(bins[n_covered].items)
            else:
                break
        
        for trigger in bins[n_covered].items:
            if verbose >= 2:
                print('trigger:', trigger, file=sys.stderr)
            trigger_score = bins[n_covered].items[trigger]

            for e in evocab:
                js = {}
                for j in range(n):
                    if trigger.coverage & 1<<j == 0 and fwords[j] in t[e]:
                        js[j] = t[e][fwords[j]]
                lmscore = math.log(langmodel.prob(trigger.lmstate, e)) + word_reward
                q = (trigger.lmstate + (e,))[-lmorder+1:]
                for r in range(1, max_fertility+1):
                    for sub_js in itertools.combinations(js, r):
                        c = trigger.coverage
                        s = trigger_score + lmscore
                        for j in sub_js:
                            c |= 1<<j
                            s += js[j]
                        bins[n_covered+r].add(Item(c, q, trigger), s)

            # try completing the translation
            s = trigger_score
            for j in range(n):
                if trigger.coverage & 1<<j == 0:
                    # even unknown words have t(f|NULL) > 0
                    s += t['NULL'].get(fwords[j], -100)
            s += math.log(langmodel.prob(trigger.lmstate, '</s>'))
            goals.add(Item((1<<n)-1, ('</s>',), trigger), s)

    if len(goals.items) == 0:
        print("no ! that 's impossible !", flush=True)
        if verbose:
            print('no translation found', file=sys.stderr)
    elif len(goals.items) == 1:
        goal, = goals.items
        ewords = []
        item = goal
        while item is not None:
            ewords.append(item.lmstate[-1])
            item = item.pointer
        ewords.reverse()
        ewords = ewords[1:-1] # <s> and </s>
        print(' '.join(ewords), flush=True)
        if verbose:
            print('output sentence:', ' '.join(ewords), file=sys.stderr)
    else:
        assert False
