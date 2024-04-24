from django.contrib import admin
from .models import *
# Register your models here.

@admin.register(Products)
class ProductAdmin(admin.ModelAdmin):
    ordering = ["id"]
    search_fields   = ['title']


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    autocomplete_fields = ['product']

    

admin.site.register(Cart)