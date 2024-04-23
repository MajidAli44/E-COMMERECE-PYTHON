from django.db import models
from uuid import uuid4
from authentication.models import User
from django.core.validators import MinValueValidator

class Products(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=30)
    unit_price = models.DecimalField(
        max_digits=6, 
        decimal_places=2,
        validators=[MinValueValidator(1)]                            
        )
    inventory = models.IntegerField(
        validators=[MinValueValidator(1)]
    )
    image = models.ImageField(upload_to='product/')
    description = models.TextField()
    last_update = models.DateTimeField(auto_now=True)
 
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


# class ProductsInformation(models.Model):
#     id = models.AutoField(primary_key=True)
#     gender = models.CharField(max_length=20)
#     mastercategory = models.CharField(max_length=50)
#     subcategory = models.CharField(max_length=50)
#     articletype = models.CharField(max_length=50)
#     basecolour = models.CharField(max_length=50)
#     season = models.CharField(max_length=20)
#     year = models.IntegerField()  # Using IntegerField instead of YearField
#     usage = models.CharField(max_length=50)
#     productdisplayname = models.CharField(max_length=100)
#     filename = models.CharField(max_length=255)
#     link = models.CharField(max_length=255)
#     file_found = models.BooleanField(default=False)  # Adjusted the default value

#     class Meta:
#         db_table = 'products_information'
 
    

        