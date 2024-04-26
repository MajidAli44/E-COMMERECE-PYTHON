from rest_framework import serializers
from .models import *
from authentication.serializers import UserSerializer
from Eshop import settings

class ProductReadSerializer(serializers.ModelSerializer):
    # cart_set= serializers.SerializerMethodField()
    image = serializers.SerializerMethodField()
    class Meta:
        model = Products
        fields = ['id','title','unit_price','image','description']

    
    def get_image(self,obj):
        if obj.image:
            if "http://assets.myntassets.com" in obj.image.name:
                return obj.image.name
            return "{0}{1}".format(settings.MEDIA_URL, obj.image.name)
        return ""
    
    # def get_cart_set(self,obj):
    #     cart_products = obj.cart.all()
    #     user = self.context['request'].user.id
    #     cart_data = []
    #     for cart_product in cart_products:
    #         if cart_product.user.id == user:
    #             cart_data.append({
    #                 'id': cart_product.id,
    #                 'product_id': cart_product.product.id,
    #                 'user_id': cart_product.user.id
    #             })
        
    #     return cart_data
    
class ProductWriteSerializer(serializers.ModelSerializer):
    cart_set= serializers.SerializerMethodField()
    
    class Meta:
        model = Products
        fields = ['title','unit_price','image','description','cart_set']

class CartSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cart
        fields = ['id','product','quantity','user']
        

class OrderReadSerializer(serializers.ModelSerializer):
    product = ProductReadSerializer()  # Nested Serializer
    user = UserSerializer()
    class Meta:
        model = Order
        fields = ('id','user','billing','price','product','quantity','date')

class OrderWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = '__all__'      