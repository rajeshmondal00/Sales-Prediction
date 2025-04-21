from django.contrib import admin
from django.urls import path
from sales import views

urlpatterns = [
    path('', views.home, name='home'),
    path('forecasts/', views.forecast_view, name='forecast-view'),
]
