from django.db import models
from django.utils import timezone

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
	function = models.CharField(max_length=500, default="")
	user = models.CharField(max_length=200, default = "Default")
	created = models.DateTimeField(default= timezone.now())
	last_active = models.DateTimeField(auto_now=True)
	father = models.ManyToManyField('self', default=None)
	mother = models.ManyToManyField('self', default=None)
	alive = models.BooleanField(default = True)
	def __str__(self):
		return "{0}_{1}_{2}_{3}".format(self.user, self.population, self.generation, self.member)
	   
	@property
	def mean_fitness(self):
		return self.gallery_fitness/max(self.seen, 1)
	
	

