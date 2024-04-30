from rest_framework.routers import DefaultRouter
from django.urls import path,include
from .views import *
from django.conf import settings
from django.conf.urls.static import static
from .views import predict_price_view

router = DefaultRouter()
router.register(r'', ProductVIewSet, basename='products')
router.register(r'order/', OrderViewSet, basename="Orders")
router.register(r'carts', CartViewSet, basename="carts")


urlpatterns = [

    path('cart/', CartPageView, name='cart_page'),
    path('checkout/', CheckoutPageView, name='checkout_page'),
    path('predict/', predict_price_view, name='predict_price_view'),
    path('recommended/products/<user_id>', Recommend_product, name='predict_price_view'),
    path('listing/', OrderCreation.as_view(), name='listing'),
    path('retrieve/<pk>', ProductRetrieve.as_view(), name='product_retrieve'),
    path('history', UserHistoryCreate.as_view(), name='user_history'),
    path('review', ReviewView.as_view(), name='products_review'),
    path('', include(router.urls))
]