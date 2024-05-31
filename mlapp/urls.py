# mlapp/urls.py

from django.urls import include, path
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    path('api/datasets/', views.get_datasets, name='get_datasets'),
    path('api/log-dataset/', views.log_selected_dataset, name='log_selected_dataset'),
    path('api/models/<str:problem_type>/', views.get_models, name='get_models'),
    path('api/train-model/', views.train_model, name='train_model'),
    path('api/log-results/', views.log_model_results, name='log_model_results'),
    path('api/models/<str:model_name>/parameters/', views.get_model_parameters, name='get_model_parameters'),
]