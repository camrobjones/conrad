import conrad.GA as GA
from conrad.models import Artist
import io
import EDS.settings as settings
import boto
from background_task import background
import logging

logger = logging.getLogger('background_task')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('background_task.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

@background(schedule=1)
def generate_image(artist_id):
    i = Artist.objects.get(pk=artist_id)
    print("Generating {} in the background".format(i))
    logger.info("Generating {} in the background".format(i))
    if i.image == '':
        im = GA.display(i.genome)
        out = "{}.PNG".format(i)
        im.save('media/'+out)
        i.image = out
        i.save()