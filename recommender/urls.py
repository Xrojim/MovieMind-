"""
URL Configuration for Movie Recommendation System
"""
from django.urls import path, reverse
from django.http import HttpResponseRedirect
from . import views

app_name = 'recommender'

urlpatterns = [
    # Main views
    path('', views.main, name='main'),
    path('results/', views.result_view, name='results'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('movie/<int:movie_idx>/', views.movie_detail, name='movie_detail'),
    path('watchlist/', views.watchlist_view, name='watchlist'),
    path('watchlist/add/', views.add_to_watchlist, name='add_to_watchlist'),
    path('watchlist/remove/<int:item_id>/', views.remove_from_watchlist, name='remove_from_watchlist'),
    path('watchlist/recommend/', views.recommend_from_watchlist, name='recommend_from_watchlist'),
    path('history/', views.history_view, name='history'),
    path('history/recommend/', views.recommend_from_history, name='recommend_from_history'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('profile/', lambda request: HttpResponseRedirect('/analytics/')),
    
    # API endpoints
    path('api/search/', views.search_movies, name='search_movies'),
    path('api/model-status/', views.model_status, name='model_status'),
    path('api/health/', views.health_check, name='health_check'),
]
