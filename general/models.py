from django.contrib.auth.models import AbstractUser
import re
from django.contrib.auth import get_user_model

class User(AbstractUser):
    @staticmethod 
    def generate_username():
        guests = [a for a in get_user_model().objects.all() if re.match('guest[0-9]+', a.username)]
        maxuserno = max([int(re.match('guest([0-9]+)', g.username).group(1)) for g in guests] + [0])
        return 'guest' + str(maxuserno + 1)
