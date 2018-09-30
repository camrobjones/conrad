
# Create your views here.

from django.http import HttpResponseRedirect
from .models import Artist
from django.shortcuts import render
import conrad.GA as GA
import conrad.image as image
import conrad.conrad_utils as conrad_utils
import EDS.storage_utils as storage_utils
import datetime
from django.utils import timezone
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
import re
import random
import csv
from lazysignup.decorators import allow_lazy_user
from lazysignup.utils import is_lazy_user
import math

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth import views as auth_views

from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect

from django.core.files.storage import default_storage

from django.urls import reverse

from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.shortcuts import render, redirect
from django.utils.translation import ugettext as _



def home(request):
	userlist = list(set([p.user for p in  Artist.objects.all()]))
	return render(request, 'conrad/home.html', {'userlist':userlist})


def conrad_about(request):
	return render(request, 'conrad/about_conrad.html')


def main_home(request):
	return render(request, 'conrad/main_home.html')


def about(request):
	return render(request, 'conrad/main_about.html')


def privacy_policy(request):
	return render(request, 'conrad/privacy.html')

def data(request):
	if is_lazy_user(request.user):
		request.session.guest = True
		request.session.modified = True
	return render(request, 'conrad/data.html')

def learn(request):
	return render(request, 'conrad/learn.html')

def resources(request):
	return render(request, 'conrad/resources.html')

def contact(request):
	return render(request, 'conrad/contact.html')


def howto(request):
	next = request.GET.get('next')
	newpop = request.GET.get('newpop')
	return render(request, 'conrad/how_to.html', {'next':next, 'newpop':newpop})

def profile(request):
	if not request.user.is_authenticated:
		return HttpResponseRedirect('/accounts/login/')
	if is_lazy_user(request.user):
		return HttpResponseRedirect('/accounts/login/')
	else:
		return render(request, 'conrad/profile.html')


def newgen(request, population, gen, username):
	pop = Artist.objects.filter(user=username, population=population, global_pop=False, generation=gen)
	fitness = round(sum([a.fitness for a in pop]) / len(pop), 2)
	diversity = conrad_utils.genetic_diversity(pop)
	top_image = pop.order_by('-fitness', '-created')[0]
	if not top_image.image500:
		im = GA.display(top_image.genome, size=(500,500))
		out = "{}_500.PNG".format(top_image)
		storage_utils.upload(im, 'media/'+out)
		top_image.image500 = out
		top_image.save()

	return render(request, 'conrad/end_gen.html', {'population':population, 'generation':gen, 'top_image':top_image,
											'fitness':fitness, 'diversity':diversity})


# stolen from https://simpleisbetterthancomplex.com/tutorial/2017/02/18/how-to-create-user-sign-up-view.html
class SignUpForm(UserCreationForm):
	# def __init__(self):
	# 	super(SignUpForm, self).__init__()
	# 	self.password1.help_text = 'Must contain at least 8 characters.'

	first_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
	last_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
	email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')


	class Meta:
		model = User
		fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2', )


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        form.fields['password1'].help_text = 'Must contain at least 8 characters.'
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()
        form.fields['password1'].help_text = 'Must contain at least 8 characters.'
    return render(request, 'EDS/signup.html', {'form': form})




@allow_lazy_user
def userhome(request):
	username = request.user
	if is_lazy_user(request.user):
		request.session.guest = True

	pop_list = list(set([p.population for p in Artist.objects.filter(user=username, global_pop=False)]))
	if pop_list:
		pops = [Artist.objects.filter(user=username, population = p, global_pop=False) for p in pop_list]
		top_images = [p.filter(seen__gte=0).order_by('-fitness', '-created')[0] for p in pops]
		gens = [max([a.generation for a in p]) for p in pops]
		la = [max([a.last_active for a in p]) for p in pops]
		fitnesses = [round(sum([a.fitness for a in p.filter(seen__gte=0)])/
					len([a.fitness for a in p.filter(seen__gte=0)]), 2) for p in pops]
		varieties = [conrad_utils.genetic_diversity(pops[i].filter(generation=gens[i])) for i in range(len(pops))]
		no_to_go = [(len(pops[i].filter(generation=gens[i], seen__lte=0))) for i in range(len(pops))]
		for i in top_images:
			if i.image == '':
				im = GA.display(i.genome)
				out = "{}.PNG".format(i)
				storage_utils.upload(im, 'media/' + out)
				i.image = out
			i.save()
		top_images = [i.image.url for i in top_images]
		user_pops = [[pop_list[i], top_images[i], gens[i], fitnesses[i], varieties[i], no_to_go[i], la[i]] for i in range(len(pop_list))]
		new_pop = max(pop_list) + 1
	else:
		user_pops = []
		new_pop = 0

	context = {'user_pops':user_pops, 'username':username, 'new_pop':new_pop}
	return render(request, 'conrad/userhome.html', context)


@allow_lazy_user
def globalhome(request):
	username = request.user
	if is_lazy_user(request.user):
		request.session.guest = True

	pop = Artist.objects.filter(global_pop=True, alive=True)
	top_image = sorted(pop, key=lambda x: ((x.mean_fitness/10)**2) * (math.log(max(1, x.seen))+1), reverse=True)[0]
	gen = max([a.generation for a in pop])
	users = len(set([a.user for a in pop]))
	fits = [a.mean_fitness for a in pop.filter(seen__gte=0)]
	fitness = round(sum(fits) / len(fits), 2)
	variety = conrad_utils.genetic_diversity(pop)

	if top_image.image500 == '':
		im = GA.display(top_image.genome, size=(500, 500))
		out = "{}_500.PNG".format(top_image)
		storage_utils.upload(im, 'media/' + out)
		top_image.image500 = out
		top_image.save()

	top_image = top_image.image500.url

	pop_data = [top_image, gen, fitness, variety, users]

	context = {'pop_data':pop_data, 'username':username}
	return render(request, 'conrad/globalhome.html', context)


@allow_lazy_user
def local(request, population):
	username = str(request.user)
	if is_lazy_user(request.user):
		request.session.guest = True
	if username == 'global':
		return global_game(request, population)

	for key in request.POST:
	    print(key)
	    value = request.POST[key]
	    print(value)
	try:
		artists = request.POST.getlist('artist[]')
		scores = []
		for i in range(1,6):
			scores.append(request.POST.get('score[{}]'.format(i)))
		print(scores)



	except (KeyError, Artist.DoesNotExist):
		pass
	else:
		for i in range(len(artists)):
			a = Artist.objects.get(pk=artists[i])
			a.fitness = scores[i]
			a.gallery_fitness = scores[i]
			a.seen = True
			a.last_active = timezone.now()
			a.voters.add(request.user)
			a.save()

	if not Artist.objects.filter(population=population, user=username, global_pop=False):
		conrad_utils.newpop(30, 30, population, username)
		return HttpResponseRedirect(reverse('conrad:how_to') + '?next={}'.format(reverse('conrad:local', args = [population])) + '&newpop=True')
	else:
		currentpop = Artist.objects.filter(seen=False, population=population, user=username, global_pop=False)
		if not currentpop:
			maxgen = max([f.generation for f in Artist.objects.filter(population=population, user=username, global_pop=False)])
			conrad_utils.newgen(Artist.objects.filter(population=population, generation=maxgen, user=username, global_pop=False))
			return newgen(request, population, maxgen, username)

	top_images = Artist.objects.filter(seen=False, population=population, user=username, global_pop=False).order_by('member')[:5]
	for i in top_images:
		if i.image == '':
			im = GA.display(i.genome)
			out = "{}.PNG".format(i)
			storage_utils.upload(im, 'media/'+out)
			i.image = out
		i.save()
	context = {'top_images': top_images, 'username':username, 'population':population}

	return render(request, 'conrad/local.html', context)

@allow_lazy_user
def global_game(request, population):
	username = str(request.user)

	if is_lazy_user(request.user):
		request.session.guest = True
		request.session.save()
		request.session.modified = True

	if not Artist.objects.filter(user=username):
		if 'howto' not in request.session:
			request.session['howto'] = True
			return HttpResponseRedirect(reverse('conrad:how_to') + '?next={}'.format(reverse('conrad:global_game', args = [0])))
	try:
		artists = request.POST.getlist('artist[]')
		scores = []
		for i in range(1,6):
			scores.append(request.POST.get('score[{}]'.format(i)))
		print(scores)

	except (KeyError, Artist.DoesNotExist):
		pass
	else:
		for i in range(len(artists)):
			a = Artist.objects.get(pk=artists[i])
			a.fitness += int(scores[i])
			a.gallery_fitness += int(scores[i])
			a.seen += 1
			a.last_active = timezone.now()
			a.voters.add(request.user)
			a.save()
			livepop = Artist.objects.filter(alive=True, population=population, global_pop=True)
			conrad_utils.global_mate(a, livepop, username)

	while len(Artist.objects.filter(alive=True, population=population, global_pop=True)) > 200:
		candidate = conrad_utils.cull(Artist.objects.filter(alive=True, population=population, global_pop=True))
		victim = Artist.objects.get(pk=candidate.id)
		victim.alive=False
		victim.save()

	alive_pop = Artist.objects.filter(alive=True, population=population, global_pop=True)
	maxgen = max([a.generation for a in alive_pop])
	top_images_temp = alive_pop.order_by('last_active')[:20]

	top_images = random.sample(list(top_images_temp), 5)
	for i in top_images:
		if i.image == '':
			im = GA.display(i.genome)
			out = "{}.PNG".format(i)
			storage_utils.upload(im, 'media/'+out)
			i.image = out
		i.save()
	context = {'top_images': top_images, 'population':population, 'username':username, 'maxgen':maxgen}

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
		a.voters.add(request.user)
		a.gallery_voters.add(request.user)
		a.save()

	n = timezone.now()

	if time == 'alltime':
		time_filtered = Artist.objects.all()
	else:
		time_dict = {'day':1, 'week':7, 'month':30, 'year':365}
		time_threshold = n - datetime.timedelta(time_dict[time])
		time_filtered = Artist.objects.filter(created__gte=time_threshold)

	if group == 'local':
		filtered = time_filtered.filter(global_pop=False)
	elif group == 'global':
		filtered = time_filtered.filter(global_pop=True)
	elif group == 'all':
		filtered = time_filtered
	else:
		filtered = time_filtered.filter(user=group)

	filtered = filtered.filter(seen__gte=1)

	if sort == 'top':
		top_images = filtered.order_by('-gallery_fitness')
	elif sort == 'average':
		top_images = sorted(filtered, key=lambda x: x.mean_fitness, reverse=True)
	elif sort == 'best':
		top_images = sorted(filtered, key=lambda x: ((x.mean_fitness/10)**2) * (math.log(max(1, x.seen))+1), reverse=True)
	elif sort == 'recent':
		top_images = sorted(filtered.filter(fitness__gte=9), key=lambda x: x.created, reverse=True)



	top_images = top_images[index:index+10]

	if index < 10:
		pindex = 0
	else:
		pindex = index - 10

	lindex = (len(filtered) // 10) * 10

	nindex = index + 10

	for i in top_images:
		if i.image == '':
			im = GA.display(i.genome)
			out = "{}.PNG".format(i)
			storage_utils.upload(im, 'media/'+out)
			i.image = out
			i.save()

	voted = [i+1 for i in range(len(top_images)) if top_images[i].voters.values().filter(username=request.user.username).count()]
	context = {'top_images': top_images, 'sort':sort, 'time':time, 'group':group,
				'index':index, 'nindex':nindex, 'pindex':pindex, 'lindex':lindex, 'voted':voted}

	return render(request, 'conrad/gallery.html', context)




def inspect(request, id):
	a = Artist.objects.get(pk=id)
	if a.mother.values().filter(generation=(a.generation - 1)):
		mother = Artist.objects.get(pk=a.mother.values().filter(generation=(a.generation - 1))[0]['id'])
		conrad_utils.check_generated(mother)
	else:
		mother = ''
	if a.father.values().filter(generation=(a.generation - 1)):
		father = Artist.objects.get(pk=a.father.values().filter(generation=(a.generation - 1))[0]['id'])
		conrad_utils.check_generated(mother)
	else:
		father = ''

	fchildren = [Artist.objects.get(pk=c['id']) for c in a.father.values().filter(generation=(a.generation + 1))]
	mchildren = [Artist.objects.get(pk=c['id']) for c in a.mother.values().filter(generation=(a.generation + 1))]
	if not a.image500:
		im = GA.display(a.genome, size=(500,500))
		out = "{}_500.PNG".format(a)
		storage_utils.upload(im, 'media/'+out)
		a.image500 = out
		a.save()

	for c in [fchildren, mchildren]:
		for i in c:
			conrad_utils.check_generated(i)

	children = mchildren + fchildren
	context = {'artist':a, 'mother':mother, 'father':father, 'children':children}
	return render(request, 'conrad/inspect.html', context)


def create_global_pop(request):
	if not request.user.is_superuser:
		return redirect('/admin/')
	all = Artist.objects.filter(global_pop=True, alive=True)
	pop_nos = list(set([x.population for x in all]))
	pops = {}
	for i in pop_nos:
		pops[i] = len(all.filter(population=i))


	if request.POST.get('doit'):
		conrad_utils.newgpop(200, 30, 0, 'global')
		success = 'True'
	else:
		success = False
	return render(request, 'conrad/ngp.html', {'success':success, 'pops':pops})


def data_download(request):
	# Create the HttpResponse object with the appropriate CSV header.
	response = HttpResponse(content_type='text/csv')
	response['Content-Disposition'] = 'attachment; filename="GA_conrad_{}.csv"'.format(
		datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	artists = Artist.objects.all()

	writer = csv.writer(response)
	writer.writerow(['id', 'genome', 'fitness', 'gallery_fitness', 'generation', 'population',
					 'member', 'seen', 'function', 'user', 'created', 'last_active', 'alive', 'global_pop'])
	for a in artists:
		writer.writerow([a.id, a.genome, a.fitness, a.gallery_fitness, a.generation, a.population,
						 a.member, a.seen, a.function, a.user, a.created, a.last_active, a.alive, a.global_pop])

	return response

def sandbox(request):
	genome1 = request.GET.get('image1')
	genome2 = request.GET.get('image2')
	if genome1:
		genome1 = genome1.upper()
		if re.search('[^ATCG]', genome1):
			error_message = "Genomes can only contain 'A's, 'T's, 'C's, and 'G's."
			return render(request, 'conrad/sandbox.html', {'image1':'', 'function1':'', 'error_message':error_message})
		if len(genome1) < 30 or len(genome1) > 90:
			error_message = "Genomes must be between 30 and 90 characters"
			return render(request, 'conrad/sandbox.html', {'image1':'', 'function1':'', 'error_message':error_message})
		else:
			image1 = 'media/' + genome1 + '.png'
			if not storage_utils.check_exists(genome1 + 'png'):
				im1 = GA.display(genome1)
				storage_utils.upload(im1, image1)
			image1 = storage_utils.get_url(image1)
			function1 = image.print_gene(genome1, 0, 1)

			if genome2:
				genome2 = genome2.upper()
				if re.search('[^ATCG]', genome2):
					error_message = "Genomes can only contain 'A's, 'T's, 'C's, and 'G's."
					return render(request, 'conrad/sandbox.html',
								  {'error_message': error_message})
				if len(genome1) < 30 or len(genome1) > 90:
					error_message = "Genomes must be between 30 and 90 characters"
					return render(request, 'conrad/sandbox.html', {'image1':'', 'function1':'', 'error_message':error_message})
				else:
					image2 = 'media/' + genome2 + '.png'
					if not storage_utils.check_exists(genome2 + 'png'):
						im2 = GA.display(genome2)
						storage_utils.upload(im2, image2)
					image2 = storage_utils.get_url(image2)
					function2 = image.print_gene(genome2, 0, 1)
					offspring = request.GET.get('offspring')
					if offspring == "True":
						random.seed(timezone.now())
						genome3 = GA.crossover(genome1, genome2)
						image3 = 'media/' + genome3 + '.png'
						if not storage_utils.check_exists(genome3 + 'png'):
							im3 = GA.display(genome3)
							storage_utils.upload(im3, image3)
						image3 = storage_utils.get_url(image3)
						function3 = image.print_gene(genome3, 0, 1)


						return render(request, 'conrad/sandbox.html',
						{'image1': image1, 'function1': function1, 'genome1': genome1,
						'image2': image2, 'function2': function2, 'genome2': genome2,
						'image3': image3, 'function3': function3, 'genome3': genome3})
					else:
						return render(request, 'conrad/sandbox.html',
								{'image1': image1, 'function1': function1, 'genome1': genome1,
								'image2': image2, 'function2': function2, 'genome2': genome2})

			else:
				return render(request, 'conrad/sandbox.html',
							{'image1': image1, 'function1': function1, 'genome1': genome1})
	else:
		return render(request, 'conrad/sandbox.html', {'image1':'', 'function1':''})


def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, _('Your password was successfully updated!'))
            return redirect('accounts:change_password')
        else:
            messages.error(request, _('Please correct the error below.'))
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'registration/change_password.html', {
        'form': form
    })



