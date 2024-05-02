from django.db import models

# Create your models here.

class ProductReviewHistory(models.Model):
    product = models.ForeignKey("products.Products",on_delete=models.CASCADE)
    positive_reviews = models.PositiveIntegerField()
    negative_reviews = models.PositiveIntegerField
    total_reviews = models.PositiveIntegerField
