# Generated by Django 4.2.11 on 2024-04-26 06:26

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('products', '0003_products_articletype_products_basecolour_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='products',
            name='basecolour',
        ),
        migrations.RemoveField(
            model_name='products',
            name='inventory',
        ),
        migrations.RemoveField(
            model_name='products',
            name='last_update',
        ),
    ]
