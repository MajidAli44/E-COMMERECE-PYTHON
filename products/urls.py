from rest_framework.routers import DefaultRouter
from django.urls import path,include
from .views import *
from django.conf import settings
from django.conf.urls.static import static
from .views import predict_price_view

router = DefaultRouter()
router.register(r'', ProductVIewSet, basename='products')
router.register(r'order', OrderViewSet, basename="Orders")


urlpatterns = [

    path('cart/', CartPageView, name='cart_page'),
    path('checkout/', CheckoutPageView, name='checkout_page'),
    path('predict/', predict_price_view, name='predict_price_view'),
    path('token', check_token, name='predict_price_view'),
    path('recommended/products/<user_id>', recommend_products, name='predict_price_view'),
    path('', include(router.urls)),
]