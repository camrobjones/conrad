import io
import boto
from django.conf import settings
def upload(pil_image, s3name):
    in_mem_file = io.BytesIO()
    pil_image.save(in_mem_file, format='PNG')
    S3Bucket = 'eds-conrad'
    s3 = boto.connect_s3(settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, host='s3.eu-west-2.amazonaws.com')
    bucket = s3.get_bucket(S3Bucket)
    key = bucket.new_key('media/{}.PNG'.format(s3name))
    in_mem_file.seek(0)
    key.set_contents_from_file(in_mem_file)
    key.set_acl('public-read')