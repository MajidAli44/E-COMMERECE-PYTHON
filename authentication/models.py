from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import AbstractUser, UserManager
from django.core.validators import RegexValidator
from .managers import CustomUserManager

class User(AbstractUser):
    username = None
    email = models.EmailField(unique=True)

    phone_no = models.CharField( max_length=15 )
    image = models.ImageField( upload_to='user_images/' )
    street = models.CharField( max_length=50 )
    city = models.CharField( max_length=50 )
    country = models.CharField( max_length=50 )
    postcode = models.CharField( max_length=50 )
    birth_date = models.DateField( null=True )
    

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email