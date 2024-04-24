from django.db import models
from uuid import uuid4
from authentication.models import User
from django.core.validators import MinValueValidator
from Eshop import settings

class Products(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=255)
    unit_price = models.DecimalField(
        max_digits=6, 
        decimal_places=2,
        validators=[MinValueValidator(1)]                            
        )
    inventory = models.IntegerField(
        validators=[MinValueValidator(1)]
    )
    image = models.ImageField(upload_to='product/',max_length=255)
    description = models.TextField()
    last_update = models.DateField(auto_now=True)
 
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