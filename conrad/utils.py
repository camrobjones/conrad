import conrad.GA as GA
from .models import Artist
import io
import EDS.settings as settings
import boto
from background_task import background
import conrad.tasks as tasks
import random

@background(schedule=1)
def generate_image(artist_id):
    i = Artist.objects.get(pk=artist_id)
    print("Generating {} in the background".format(i))
    if i.image == '':
        im = GA.display(i.genome)
        out = "{}.PNG".format(i)
        im.save('media/'+out)
        i.image = out
        i.save()

def newpop(popsize, genlength, pop, user):
    for i in range(popsize):
        a = Artist(population = pop, generation = 0, fitness = 0, member = i,
        seen = False, image = None, genome = GA.birth(genlength), user = user)
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
    return(p1.genome[:sp1]+p2.genome[sp1:sp2]+p1.genome[sp2:])


def mutate(agent, m_rate):
    a = list(agent)
    for i in range(len(agent)):
        if random.random() < m_rate:
            a[i] = random.choice('ATCG')
    return ''.join(a)

#grid = [(i,j) for i,j in range(7)]

        
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
        child = mutate(child, 0.1)
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
        
def global_mate(mother, population):
    popNo = mother.population
    gen = mother.generation + 1
    member = max([p.member for p in population if p.generation == gen]+ [-1]) + 1
    mf = round(sum([p.mean_fitness for p in population]))
    father = select_parent(population, mean=True)
    if len(father.genome) < len(mother.genome):
        mother, father = father, mother
    child = crossover(mother, father)
    child = mutate(child, 0.1)
    a = Artist(population = popNo, generation = gen, fitness = mf, member = member,
    seen = False, image = '', image500 = '', genome = child, user = 'global')
    a.save()
    a.mother.set([mother.id])
    a.father.set([father.id])
    a.save()
    tasks.generate_image(a.id)
    if random.random() < 0.1:
        a = Artist(population = popNo, generation = gen, fitness = mf, member = member + 1,
        seen = False, image = '', image500 = '', genome = GA.birth(30), user = 'global')
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

def upload(pil_image, s3name):
    in_mem_file = io.BytesIO()
    pil_image.save(in_mem_file, format='PNG')


    S3Bucket = 'eds-conrad'
    s3 = boto.connect_s3(settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, host='s3.eu-west-2.amazonaws.com')
    bucket = s3.get_bucket(S3Bucket)
    key = bucket.new_key('{}'.format(s3name))
    in_mem_file.seek(0)
    key.set_contents_from_file(in_mem_file)
    key.set_acl('public-read')
    
from background_task import background
from django.contrib.auth.models import User

