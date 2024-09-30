from . import views
from django.urls import path
urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('fixprice/', views.price_fix, name='price_fix'),
    path('dataSubmition/', views.commodity_data_submission, name='result'),

    path('graph/', views.graph, name='graph'),
    path('get-price-data/<str:commodity>/', views.get_price_data, name='get_price_data'),
    path('buffer-stock/', views.buffer_stock, name='buffer_stock'),
    path('update-stock/', views.update_stock, name='update_stock'),
    path('graph-data/', views.graphs_plot, name='show_graph'),

]