from rest_framework.routers import DefaultRouter
from django.urls import path,include
from .views import *
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import (
    TokenRefreshView,TokenObtainPairView
)

router = DefaultRouter()
router.register(r'user', UserViewSet, basename='users')


urlpatterns = [
    path('signupPage', Signup_Page, name='signup_page'),
    path('loginPage', Login_Page, name='login_page'),
    path('forgetpasswordPage', render_forgetpassword_page, name='forgetpassword'),
    path('profilePage', render_profile_page, name='userProfile'),
    path('token', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', logout_view, name='logout'),
    path('', include(router.urls)),
]