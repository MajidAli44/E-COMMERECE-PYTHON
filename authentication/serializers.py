from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import *

        
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email','phone_no','password','image','street','city','country','postcode','birth_date'] 
        extra_kwargs = {'password': {'write_only': True}}
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)  # Set the encrypted password
        user.save()
        return user

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['user_id'] = user.id
        token['email'] = user.email
        token['role'] = user.is_superuser
        # ...

        return token