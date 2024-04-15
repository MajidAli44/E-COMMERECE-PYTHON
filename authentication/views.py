from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import *

def Signup_Page(request):
    if request.method == "GET":
        return render(request,'signup.html')
    elif request.method == 'POST':
        serializer = UserWriteSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            user.save()
            return Response({'message': 'Record created successfully'}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def Login_Page(request):
    return render(request,'login.html')

def render_profile_page(request):
    return render(request, "profile.html")  

def render_forgetpassword_page(request):
    return render(request, "forgetpassword.html") 


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all
    
    def get_serializer_class(self):
        if self.action in ('create','update','partial_update'):
            return UserWriteSerializer
        return UserReadSerializer