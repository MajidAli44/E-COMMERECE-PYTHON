from rest_framework import serializers
from .models import *

class ProductReadSerializer(serializers.ModelSerializer):
    cart_set= serializers.SerializerMethodField()
    class Meta:
        model = Products
        fields = ['id','title','unit_price','inventory','image','description','cart_set']

    def get_cart_set(self,obj):
        cart_products = obj.cart.all()
        user = self.context['request'].user.id
        cart_data = []
        for cart_product in cart_products:
            if cart_product.user.id == user:
                cart_data.append({
                    'id': cart_product.id,
                    'product_id': cart_product.product.id,
                    'user_id': cart_product.user.id
                })
        
        return cart_data
    
class ProductWriteSerializer(serializers.ModelSerializer):
    cart_set= serializers.SerializerMethodField()
    
    class Meta:
        model = Products
        fields = ['title','unit_price','inventory','image','description','cart_set']

class CartSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cart
        fields = ['id','product','quantity','user']
        

class OrderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = ['id','product','price','quantity']