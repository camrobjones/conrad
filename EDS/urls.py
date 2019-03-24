"""EDS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path
from django.conf import settings
from lazysignup.views import convert

from conrad import views

urlpatterns = [
    path('conrad/', include('conrad.urls')),    
    path('admin/', admin.site.urls),
    path('', views.main_home, name = 'home'),
    path('about/', views.about, name = 'about'),

    path('accounts/login/',  auth_views.LoginView.as_view(
        template_name='EDS/login.html'), name='login'),

    path('accounts/reset/', auth_views.PasswordResetView.as_view(
        template_name='EDS/password_reset.html'),
         name='password_reset'),

    path('accounts/logout/',  auth_views.LogoutView.as_view(
        template_name='EDS/logout.html'), name='logout'),

    path('accounts/profile/', views.profile, name='profile'),
    path('accounts/signup/', views.signup, name='signup'),

    path('convert/', convert, name = 'lazysignup_convert'),
     #    include('lazysignup.urls')),
    path('accounts/privacy_policy/', views.privacy_policy,
     name='privacy_policy'),

    path('accounts/password/', views.change_password,
     name='change_password'),

    path('password_reset/',
         auth_views.PasswordResetView.as_view(
            template_name='EDS/password_reset')),

    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='registration/password_reset_done.html'), 
    name='password_reset_done'),

    path('reset/<uidb64>[0-9A-Za-z_\-]+/<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20}/',
        auth_views.PasswordResetConfirmView.as_view( 
        template_name='registration/password_reset_confirm.html'),
    name='password_reset_confirm'),

    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='registration/password_reset_complete.html'),
        name='password_reset_complete'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
