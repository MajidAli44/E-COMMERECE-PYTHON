from django.urls import path
from .views import *


urlpatterns = [
    path("",AdminPage, name="admin_page"),
    path("product",ProductPage, name="product_page"),
    path("order",OrderPage, name="order_page"),
    path("users",UserPage, name="user_page"),
    path("sentiment",predict_sentiment, name="sentiment_page"),
    
]
