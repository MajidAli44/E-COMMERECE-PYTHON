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
        
class UserHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = UserHistory
        fields = '__all__'    
        
class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = ['product','name','description','rating']