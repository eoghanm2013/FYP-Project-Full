import json

from django.contrib.sites import requests
from django.views.generic import TemplateView
from newsapi import NewsApiClient
from .models import Question, Choice, AggData, Predictions, CompanyInfo
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import StockSerializer, PredictionSerializer, CompanyInfoSerializer
from django.http import JsonResponse


## REST API

@api_view(['GET'])
def stock_collection(request):
    if request.method == 'GET':
        brands = AggData.objects.all()
        serializer = StockSerializer(brands, many=True)
        return Response(serializer.data, template_name='index.html')


@api_view(['GET'])
def stock_element(request, brand):
    try:
        brand = AggData.objects.filter(brand=brand)
    except AggData.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = StockSerializer(brand, many=True)
        return Response(serializer.data, template_name='index.html')


def get_data(requests, *args, **kwargs):
    model = AggData
    data = AggData.objects.all()
    serializer = StockSerializer(data, many=True)
    return JsonResponse(serializer.data, safe=False)


@api_view(['GET'])
def prediction_collection(request):
    if request.method == 'GET':
        predictions = Predictions.objects.all()
        serializer = PredictionSerializer(predictions, many=True)
        return Response(serializer.data, template_name='index.html')


@api_view(['GET'])
def prediction_element(request, brand):
    try:
        brand = Predictions.objects.filter(brand=brand)
    except Predictions.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = PredictionSerializer(brand, many=True)
        return Response(serializer.data, template_name='index.html')


@api_view(['GET'])
def company_element(request, Brand):
    try:
        brand = CompanyInfo.objects.filter(Brand=Brand)
    except CompanyInfo.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = CompanyInfoSerializer(brand, many=True)
        return Response(serializer.data, template_name='index.html')


def get_predictions(requests, *args, **kwargs):
    model = Predictions
    data = Predictions.objects.all()
    serializer = PredictionSerializer(data, many=True)
    return JsonResponse(serializer.data, safe=False)


class CompanyView(generic.ListView):
    context_object_name = 'brands'
    template_name = 'polls/index.html'

    def get_queryset(self):
        """Return all for now"""
        return AggData.objects.filter(brand='Activision')


def profile(request, brandname):
    brand = AggData.objects.filter(brand=brandname)
    context = dict(brands=brand)
    return render(request, 'polls/index.html', context)


# ----------------------------- DJANGO BOOTCAMP SHIT GOES IN HERE ---------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------------------------------------------



from django.shortcuts import render
from newsapi import NewsApiClient
import requests


    # Create your views here.
def Index(request, brand):

    url = "https://newsapi.org/v2/everything?q="+ brand + "&pageSize=6&apiKey=25a911c33c9747ad965eae6348f799f8"

    response = requests.get(url)
    content_from_internet = json.loads(response.content)
    context = {
        'data': content_from_internet,
    }
    return render(request, 'polls/index.html', context)

def Basic(request):
    return render(request,'basic.html' )

def About(request):
    return render(request, 'polls/about.html')
