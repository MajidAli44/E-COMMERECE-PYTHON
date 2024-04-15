from django.shortcuts import render
from rest_framework import routers, serializers, viewsets
from .serializers import *
from .models import *

# Create your views here.

def IndexPageView(request):
    return render(request, "index.html")

def CartPageView(request):
    return render(request, "cart.html")

def CheckoutPageView(request):
    return render(request, "checkout.html")

class ProductVIewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing Product instances.
    """
    queryset = Products.objects.all()
    
    def get_serializer_class(self):
        if self.action in ('create', 'update', 'partial_update',):
            return ProductWriteSerializer
        return ProductReadSerializer
    

class CartViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing Cart instances.
    """
    serializer_class = CartSerializer
    queryset = Cart.objects.all()
    
class OrderViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing Cart instances.
    """
    serializer_class = OrderSerializer
    queryset = Order.objects.all()