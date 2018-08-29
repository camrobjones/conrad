from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
import conrad.image as image

# Create your models here.

class Artist(models.Model):
	genome = models.CharField(max_length=200)
	fitness = models.IntegerField(default = 0)
	gallery_fitness = models.IntegerField(default = 0)
	image = models.ImageField(upload_to='static/conrad/images',
							   blank=True,
							   null=True,
							   )
	image500 = models.ImageField(
							   blank=True,
							   null=True,
							   )
	generation = models.IntegerField(default = 0)
	population = models.IntegerField(default = 0)
	member = models.IntegerField(default = 0)
	seen = models.IntegerField(default = 0)
	user = models.CharField(max_length=200, default = "Default")
	created = models.DateTimeField(default=timezone.now)
	last_active = models.DateTimeField(auto_now=True)
	father = models.ManyToManyField('self', default=None)
	mother = models.ManyToManyField('self', default=None)
	voters = models.ManyToManyField(User, default=None)
	gallery_voters = models.ManyToManyField(User, related_name='+', default=None)
	alive = models.BooleanField(default=True)
	global_pop = models.BooleanField(default=False)
	def __str__(self):
		return "{0}{1}_{2}_{3}_{4}".format(self.user, '_global' if self.global_pop else '', self.population, self.generation, self.member)

	@property
	def mean_fitness(self):
		return self.gallery_fitness/max(self.seen, 1)

	@property
	def function(self):
		return image.print_gene(self.genome, 0, 1)

	
	

