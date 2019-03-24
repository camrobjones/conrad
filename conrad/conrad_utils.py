import conrad.GA as GA
import conrad.image as image
from conrad.models import Artist
import io
from django.conf import settings
import boto
from background_task import background
import ga.tasks as tasks
import random
import ga.storage_utils as storage_utils
import stringdist

def newpop(popsize, genlength, pop, user):
    for i in range(popsize):
        a = Artist(population=pop, generation=0, fitness=0, member=i,
                   seen=False, image=None, genome=GA.birth(genlength), user=user)
        a.save()
        tasks.generate_image(a.id)


def newgpop(popsize, genlength, pop, user):
    for i in range(popsize):
        a = Artist(population=pop, generation=0, fitness=0, member=i,
                   seen=False, image=None, genome=GA.birth(genlength), user=user, global_pop=True)
        a.save()
        tasks.generate_image(a.id)

def select_parent(pop, mean=False):
    if mean:
        fitlist = [p.mean_fitness for p in pop]
    else:
        fitlist = [p.fitness for p in pop]
    thresh = random.random() * sum(fitlist)
    acc = 0
    for artist in pop:
        if mean:
            acc += artist.mean_fitness
        else:
            acc += artist.fitness
        if acc >= thresh:
            return artist


def crossover(p1, p2):
    sp1 = random.randrange(0, len(p1.genome))
    sp2 = random.randrange(sp1, len(p1.genome))
    if p2.fitness > p1.fitness & len(p2.genome) > len(p1.genome):
        return p1.genome[:sp1] + p2.genome[sp1:sp2] + p1.genome[sp2:] + p2.genome[len(p1.genome):]
    else:
        return p1.genome[:sp1] + p2.genome[sp1:sp2] + p1.genome[sp2:]


def mutate(agent, m_rate, g_rate):
    a = list(agent)
    for i in range(len(agent)):
        if random.random() < m_rate:
            a[i] = random.choice('ATCG')
    if random.random() < g_rate:
        a = [random.choice('ATCG') for i in range(3)] + a  #potential for genome growth
    return ''.join(a)

        
def newgen(population):
    popNo = population[0].population
    gen = population[0].generation + 1
    user = population[0].user
    population = list(population)
    random.shuffle(population)
    
    for i in range(len(population) - int(len(population)/10)):
        mother = select_parent(population)
        father = select_parent(population)
        if len(father.genome) < len(mother.genome):
            mother, father = father, mother
        child = crossover(mother, father)
        child = mutate(child, 0.1, 0.5)
        a = Artist(population = popNo, generation = gen, fitness = 0, member = i,
        seen = False, image = '', image500 = '', genome = child, user = user)
        a.save()
        a.mother.set([mother.id])
        a.father.set([father.id])
        a.save()
        tasks.generate_image(a.id)
    for i in range(int(len(population)/10)):
        a = Artist(population = popNo, generation = gen, fitness = 0, member = len(population) - int(len(population)/10) + i,
        seen = False, image = '', image500 = '', genome = GA.birth(30), user = user)
        a.save()
        tasks.generate_image(a.id)


def check_generated(i):
    if i.image == '':
        im = GA.display(i.genome)
        out = "{}.png".format(i)
        storage_utils.upload(im, 'media/' + out)
        i.image = out
        i.save()


def global_mate(mother, population, username):
    popNo = mother.population
    gen = mother.generation + 1
    member = max([p.member for p in population if p.generation == gen]+ [-1]) + 1
    mf = round(sum([p.mean_fitness for p in population]))
    father = select_parent(population, mean=True)
    if len(father.genome) < len(mother.genome):
        mother, father = father, mother
    child = crossover(mother, father)
    child = mutate(child, 0.1, 0.5)
    a = Artist(population=popNo, generation=gen, fitness=mf, member=member,
               seen=False, image='', image500='', genome=child, user=username, global_pop=True)
    a.save()
    a.mother.set([mother.id])
    a.father.set([father.id])
    a.save()
    tasks.generate_image(a.id)
    if random.random() < 0.1:
        a = Artist(population=popNo, generation=gen, fitness=mf, member=member + 1,
                   seen=False, image='', image500='', genome=GA.birth(30), user='global', global_pop=True)
        a.save()
        tasks.generate_image(a.id)


def cull(pop):
    fitlist = [1/max(p.mean_fitness, 1) for p in pop]
    thresh = random.random() * sum(fitlist)
    acc = 0
    for artist in pop:
        acc += 1/max(1, artist.mean_fitness)
        if acc >= thresh:
            return artist

def genetic_diversity(pop):
    dists = [stringdist.levenshtein_norm(pop[x].genome, pop[y].genome) for x in range(len(pop)) for y in range(x+1, len(pop))]
    variety = round(sum(dists) * 100 / len(dists))
    return variety
    

