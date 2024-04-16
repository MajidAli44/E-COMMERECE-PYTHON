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
    image = models.ImageField( upload_to='user_images/' , null=True, blank=True)
    street = models.CharField("Street Address", max_length=50 ,null=True, blank=True)
    city = models.CharField("City", max_length=50, null=True, blank=True)
    country = models.CharField("Country", max_length=50,null=True, blank=True )
    postcode = models.CharField("PostCode", max_length=50, null=True, blank=True )
    birth_date = models.DateField("Date Of Birth", null=True, blank=True )
    

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email