from django.urls import path

from . import views

app_name = "conrad"
urlpatterns = [
    path('', views.home, name='home'),
    path('gallery/<str:sort>/<str:time>/<str:group>/<int:index>/', views.gallery, name='gallery'),
    path('local/', views.userhome, name='userhome'),
    path('global/', views.globalhome, name='globalhome'),
    path('local/<int:population>/', views.local, name='local'),
    path('global/<int:population>/', views.global_game, name='global_game'),
    path('login/', views.login, name='login'),
    path('about/', views.conrad_about, name="conrad_about"),
    path('inspect/<int:id>', views.inspect, name="inspect"),
    path('ngp/', views.create_global_pop, name="ngp"),
    path('data_download/', views.data_download, name='data_download'),
    path('sandbox/', views.sandbox, name='sandbox'),
    path('data/', views.data, name='data'),
    path('learn/', views.learn, name='learn'),
    path('resources/', views.resources, name='resources'),
    path('contact/', views.contact, name='contact'),
]
