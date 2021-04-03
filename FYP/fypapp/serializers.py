from rest_framework import serializers
from fypapp.models import AggData, Predictions, CompanyInfo


class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = AggData
        fields = (
        'pos', 'neg', 'neu', 'comp', 'brand', 'day', 'ticker', 'closeprice', 'SMA5', 'EMA', 'fut1', 'fut3', 'fut5')


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Predictions
        fields = ('p_sentiment', 'p_5day', 'p_5dayEMA', 'p_EMA', 'p_sentimentEMA', 'p_realprice', 'brand')

class CompanyInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = CompanyInfo
        fields = ('MA5score','EMA5score','EMAscore','sentMAscore','Brand','CEO','Revenue','Industry','Location','Employees')