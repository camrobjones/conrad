

import random
import PIL
from PIL import Image, ImageColor
import numpy as np
import math
from itertools import chain
import pandas as pd
import conrad.image as image



def birth(popsize):
    return ''.join([random.sample('ACGT', 1)[0] for i in range(popsize)])


def select_parent(pop, fitlist):
    thresh = random.random() * sum(fitlist)
    acc = 0
    for j in range(len(pop)):
        acc += fitlist[j]
        if acc >= thresh:
            return pop[j]


def crossover(p1, p2):
    if len(p1) > len(p2):
        p1, p2 = p2, p1
    sp1 = random.randrange(0, len(p1))
    sp2 = random.randrange(sp1, len(p1))
    if len(p2) > len(p1):
        return p1[:sp1] + p2[sp1:sp2] + p1[sp2:] + p2[len(p1):]
    else:
        return p1[:sp1] + p2[sp1:sp2] + p1[sp2:]


def mutate(agent, m_rate):
    a = list(agent)
    for i in range(len(agent)):
        if random.random() < (m_rate/2):
            a[i] = random.choice('ATCG')
    if random.random() < (m_rate * 2):
        a = [random.choice('ATCG') for i in range(3)] + a  #potential for genome growth
    return ''.join(a)

def newpop(pop, fitlist, genelength):
    new_pop = []
    for i in range(len(pop) - int(len(pop)/10)):
        parent1 = select_parent(pop, fitlist)
        parent2 = select_parent(pop, fitlist)
        child = crossover(parent1, parent2)
        child = mutate(child, 0.1)
        new_pop.append(child)
    for i in range(int(len(pop)/10)):
        new_pop.append(birth(genelength))
    return(new_pop)

# Real Normalization
# def normalized(a, axis=-1, order=np.inf):
#     mins = np.array([np.min(a[:, :, x]) for x in range(3)])
#     maxes = np.array([np.max(a[:, :, x]) for x in range(3)])
#     norms = (maxes - mins)
#     norms[norms==0] = 0.0001
#     a = (a-mins)/norms
#     return a

# Not genuine normalization but the images are improved
    #To do - reconcile these two
def normalized(a, axis=-1, order=np.inf):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def display(agent, size=(200, 200)):
    random.seed(agent)
    np.random.seed(sum([ord(x) for x in agent]))
    pixels = image.read_gene(agent, 0, 1, size)
    outnorm = normalized(pixels) * 255
    im = Image.fromarray(np.uint8(outnorm))
    return im