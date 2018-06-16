from django.urls import path

from . import views

app_name = "conrad"
urlpatterns = [
        path('', views.home, name = 'home'),
		path('gallery/<str:sort>/<str:time>/<str:group>/<int:index>/', views.gallery, name = 'gallery'),
		path('local/<str:username>/', views.userhome, name = 'userhome'),
        path('local/<str:username>/<int:population>/', views.local, name = 'local'),
		path('global/<int:population>/', views.global_game, name = 'global_game'),
		path('login/', views.login, name = 'login'),
		path('about/', views.conrad_about, name = "conrad_about"),
		path('inspect/<int:id>', views.inspect, name = "inspect"),
        ]
	