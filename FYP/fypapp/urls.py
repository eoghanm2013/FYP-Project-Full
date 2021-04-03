from django.urls import path
from django.conf.urls import url
from . import views
from .views import Index, Basic, About

app_name= 'fypapp'
urlpatterns = [
    #path('', views.IndexView.as_view(), name='index'),
    #path('test', views.CompanyView.as_view(), name='test'),
    path('', views.CompanyView.as_view(), name='index'),
    path('<str:brand>/', Index, name = 'Index'),
    path('<str:brandname>/', views.profile),
    url(r'^api/predictions/$', views.prediction_collection, name='getpredictions'),
    url(r'^api/predictions/(?P<brand>.+)/$', views.prediction_element, name='getprediction_element'),
    url(r'^api/companyinfo/(?P<Brand>.+)/$', views.company_element, name='getcompany_element'),
    url(r'^api/data/$', views.get_data, name='getdata'),
    url(r'^api/brands/$', views.stock_collection),
    url(r'^api/brands/(?P<brand>.+)/$', views.stock_element),
    path('basic/basic/', Basic, name = 'Basic'),
    path('about/help/', About, name='About'),


]