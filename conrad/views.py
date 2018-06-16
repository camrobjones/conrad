from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.urls import reverse
from .models import Artist
from django.shortcuts import render
import os
import conrad.GA as GA
import conrad.utils as utils
import EDS.settings as settings
import io
import boto
import datetime
from django.utils import timezone
import pytz




def home(request):
	userlist = list(set([p.user for p in  Artist.objects.all()]))
	return render(request, 'conrad/home.html', {'userlist':userlist})

	
def conrad_about(request):
	return render(request, 'conrad/about_conrad.html')
	
def main_home(request):
	return render(request, 'conrad/main_home.html')
	
def about(request):
	return render(request, 'conrad/about.html')
	
	
	
def userhome(request, username):
	#username = request.POST.get('username')
	pop_list = list(set([p.population for p in  Artist.objects.filter(user=username)]))
	top_images = [Artist.objects.filter(user=username, population = p).order_by('-fitness', '-created')[0].image.url for p in pop_list]
	gens = [max([a.generation for a in Artist.objects.filter(user=username, population = p)]) for p in pop_list]
	la = [max([a.last_active for a in Artist.objects.filter(user=username, population = p)]) for p in pop_list]
	fitnesses = [sum([a.fitness for a in Artist.objects.filter(user=username, population = p)])/
	len([a.fitness for a in Artist.objects.filter(user=username, population = p)]) for p in pop_list]
	
	user_pops = [[pop_list[i], top_images[i], gens[i], fitnesses[i], la[i]] for i in range(len(pop_list))]
	
	if pop_list:
		new_pop = max(pop_list) + 1
	else:
		new_pop = 0
	
	context = {'user_pops':user_pops, 'username':username, 'new_pop':new_pop}
	return render(request, 'conrad/userhome.html', context)

def local(request, username, population):
	if username == 'global':
		return global_game(request, population)
	try:
		artists = request.POST.getlist('artist[]')
		scores = request.POST.getlist('score[]')

	
	except (KeyError, Artist.DoesNotExist):
		pass
	else:
		for i in range(len(artists)):
			a = Artist.objects.get(pk=artists[i])
			a.fitness = scores[i]
			a.gallery_fitness = scores[i]
			a.seen = True
			a.last_active = timezone.now()
			a.save()
	
	if not Artist.objects.filter(population=population, user=username):
		utils.newpop(50, 30, population, username)
	else:
		currentpop = Artist.objects.filter(seen=False, population=population, user=username)
		if not currentpop:
			maxgen = max([f.generation for f in Artist.objects.filter(population=population, user=username)])
			utils.newgen(Artist.objects.filter(population=population, generation=maxgen, user=username))
			
	top_images = Artist.objects.filter(seen=False, population=population, user=username).order_by('member')[:5]
	for i in top_images:
		if i.image == '':
			im = GA.display(i.genome)
			out = "{}.PNG".format(i)
			
			im.save('media/'+out)
			i.image = out
		i.save()
	context = {'top_images': top_images, 'population':population, 'username':username}
	
	
	#return HttpResponseRedirect(reverse('conrad:local', args=(username,population,)))
	return render(request, 'conrad/local.html', context)

	

def global_game(request, population):
	try:
		artists = request.POST.getlist('artist[]')
		scores = request.POST.getlist('score[]')

	
	except (KeyError, Artist.DoesNotExist):
		pass
	else:
		for i in range(len(artists)):
			a = Artist.objects.get(pk=artists[i])
			a.fitness += int(scores[i])
			a.gallery_fitness += int(scores[i])
			a.seen += 1
			a.last_active = timezone.now()
			a.save()
			livepop = Artist.objects.filter(alive=True, population=population, user='global')
			utils.global_mate(a, livepop) 
	
	while len(Artist.objects.filter(alive=True, population=population, user='global')) > 200:
		candidate = utils.cull(Artist.objects.filter(alive=True, population=population, user='global'))
		victim = Artist.objects.get(pk=candidate.id)
		victim.alive=False
		victim.save()
	
	top_images = Artist.objects.filter(alive=True, population=population, user='global').order_by('last_active')[:5]
	for i in top_images:
		if i.image == '':
			im = GA.display(i.genome)
			out = "{}.PNG".format(i)
			im.save('media/'+out)
			i.image = out
		i.save()
	context = {'top_images': top_images, 'population':population, 'username':'global'}
	
	#return HttpResponseRedirect(reverse('conrad:local', args=(username,population,)))
	return render(request, 'conrad/global.html', context)


def gallery(request, sort, time, group, index):
	try:
		a = Artist.objects.get(pk=request.POST['artist'])
		score = request.POST['score']

	except (KeyError, Artist.DoesNotExist):
		pass
	else:
		a.gallery_fitness += int(score)
		a.seen += 1
		a.save()
	
	n = timezone.now()
	
	if time == 'alltime':
		time_filtered = Artist.objects.all()
	else:
		time_dict = {'day':1, 'week':7, 'month':30, 'year':365}
		time_threshold = n - datetime.timedelta(time_dict[time])
		time_filtered = Artist.objects.filter(created__gte=time_threshold)
	
	if group == 'local':
		filtered = time_filtered.exclude(user='global')
	elif group == 'all':
		filtered = time_filtered
	else:
		filtered = time_filtered.filter(user=group)
	
	if sort == 'top':
		top_images = filtered.order_by('-gallery_fitness')
	elif sort == 'average':
		top_images = sorted(filtered, key=lambda x: x.mean_fitness, reverse=True)
	elif sort == 'best':
		top_images = sorted(filtered, key=lambda x: x.mean_fitness **2 * x.seen, reverse=True)
	
	top_images = top_images[index:index+10]
	
	if index < 10:
		pindex = 0
	else:
		pindex = index - 10
		
	nindex = index + 10
	
	for i in top_images:
		if i.image == '':
			im = GA.display(i.genome)
			out = "{}.PNG".format(i)
			im.save('media/'+out)
			i.image = out
		i.save()
	context = {'top_images': top_images, 'sort':sort, 'time':time, 'group':group, 'index':index, 'nindex':nindex, 'pindex':pindex}
	
	#return HttpResponseRedirect(reverse('conrad:local', args=(population,)))
	return render(request, 'conrad/gallery.html', context)

	
def inspect(request, id):
	a = Artist.objects.get(pk=id)
	if not a.image500:
		im = GA.display(a.genome, size=(500,500))
		out = "{}_500.PNG".format(a)
		im.save('media/'+out)
		a.image500 = out
		a.save()
	context = {'artist':a}
	return render(request, 'conrad/inspect.html', context)
	
	
def login(request):
	try:
		username = request.POST.get('username')
	except(KeyError):
		return render(request, 'conrad/home.html')
	else:
		return HttpResponseRedirect('/conrad/local/'+username+'/')

	
	