from django.db import models
from uuid import uuid4
from authentication.models import User
from django.core.validators import MinValueValidator

class products(models.Model):
    title = models.CharField(max_length=50)
    unit_price = models.DecimalField(
        max_digits=6, 
        decimal_places=2,
        validators=[MinValueValidator(1)]                            
        )
    inventory = models.IntegerField(
        validators=[MinValueValidator(1)]
    )
    image = models.ImageField(upload_to='media/')
    description = models.TextField()
    last_update = models.DateTimeField(auto_now=True)
 
    def __str__(self):
        return self.title


class cart(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    product = models.ForeignKey(products, on_delete=models.CASCADE, related_name='cart')
    quantity = models.PositiveSmallIntegerField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    
class order(models.Model):
    user = models.ForeignKey(User, on_delete=models.PROTECT, related_name='orders')
    product = models.ForeignKey(products, on_delete=models.PROTECT )
    price = models.IntegerField()
    quantity = models.PositiveSmallIntegerField()
    placed_at = models.DateTimeField(auto_now_add=True)
 
    def __str__(self):
        return self.date  
    

        