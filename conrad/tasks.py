import conrad.GA as GA
from .models import Artist
import io
import EDS.settings as settings
import boto
from background_task import background

@background(schedule=60)
def generate_image(artist_id):
    i = Artist.objects.get(pk=artist_id)
    print("Generating {} in the background".format(i))
    if i.image == '':
        im = GA.display(i.genome)
        out = "{}.PNG".format(i)
        im.save('media/'+out)
        i.image = out
        i.save()