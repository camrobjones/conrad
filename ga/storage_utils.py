from django.core.files.storage import default_storage

def upload(pil_image, s3name):
    pil_image.save(s3name)
    
def check_exists(x):
	return default_storage.exists(x)

def get_url(x):
	return '/' + x

