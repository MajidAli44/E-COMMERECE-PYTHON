from django.db import models
from uuid import uuid4
from authentication.models import User
from django.core.validators import MinValueValidator
from Eshop import settings

class Products(models.Model):
    id = models.IntegerField(primary_key=True)
    # title = models.CharField(max_length=255)
    # unit_price = models.DecimalField(
    #     max_digits=6, 
    #     decimal_places=2,
    #     validators=[MinValueValidator(1)]                            
    #     )
    # inventory = models.IntegerField(
    #     validators=[MinValueValidator(1)]
    # )
    # last_update = models.DateField(auto_now=True)
    gender = models.CharField(max_length=20)
    mastercategory = models.CharField(max_length=50)
    subcategory = models.CharField(max_length=50)
    articletype = models.CharField(max_length=50)
    # basecolour = models.CharField(max_length=50)  
    season = models.CharField(max_length=20)
    title = models.CharField(max_length=255)
    description = models.TextField(max)
    unit_price = models.IntegerField()
    image = models.ImageField(upload_to='product/',max_length=255)
 
    def __str__(self):
        return self.title

class Cart(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    product = models.ForeignKey(Products, on_delete=models.CASCADE, related_name='cart')
    quantity = models.PositiveSmallIntegerField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    
class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.PROTECT, related_name='orders')
    product = models.ForeignKey(Products, on_delete=models.PROTECT )
    price = models.IntegerField()
    quantity = models.PositiveSmallIntegerField()
    placed_at = models.DateTimeField(auto_now_add=True)
    

class UserHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='users')
    product = models.ForeignKey(Products, on_delete=models.CASCADE)
    
    
    
class Review(models.Model):
    
    PRODUCT_RATING = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    ]
    
    product = models.ForeignKey(Products,on_delete=models.CASCADE,related_name='reviews')
    name = models.CharField(max_length=255)
    description = models.TextField()
    rating  = models.IntegerField(choices=PRODUCT_RATING)
    date = models.DateField(auto_now_add=True)
    