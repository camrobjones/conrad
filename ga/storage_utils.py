import conrad.GA as GA
from conrad.models import Artist
import io
import ga.settings as settings
import boto
from background_task import background
import custom_storages


def upload(pil_image, s3name):
    in_mem_file = io.BytesIO()
    pil_image.save(in_mem_file, format='PNG')

    S3Bucket = settings.AWS_STORAGE_BUCKET_NAME
    s3 = boto.connect_s3(settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, host='s3.eu-west-2.amazonaws.com')
    bucket = s3.get_bucket(S3Bucket)
    key = bucket.new_key('{}'.format(s3name))
    in_mem_file.seek(0)
    key.set_contents_from_file(in_mem_file)
    key.set_acl('public-read')
    

def check_exists(x):

    s3 = boto.connect_s3(settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, host='s3.eu-west-2.amazonaws.com')
    S3Bucket = settings.AWS_STORAGE_BUCKET_NAME
    try:
        s3.Object(S3Bucket, 'media/' + x).load()
        return True
    except:
        return False

def get_url(x):
    return '//' + settings.AWS_S3_CUSTOM_DOMAIN + '/' + x
