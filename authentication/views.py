from django.shortcuts import render
from django.contrib.auth import logout
from django.http import HttpResponseRedirect
from django.urls import reverse
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import *
from rest_framework_simplejwt.views import TokenObtainPairView


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

def Signup_Page(request):
    if request.method == "GET":
        return render(request,'signup.html')
    elif request.method == 'POST':
        serializer = UserSerializer(data=request.data)
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
    serializer_class = UserSerializer


def logout_view(request):
    logout(request) # This logs the user out.
    # Redirect to a success page.
    print("logout called")
    return render(request,'login.html')