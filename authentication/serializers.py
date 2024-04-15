from rest_framework import serializers
from .models import *


class UserReadSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id','email','phone_no','image','street','city','country','postcode','birth_date'] 
        
class UserWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email','phone_no','image','street','city','country','postcode','birth_date'] 