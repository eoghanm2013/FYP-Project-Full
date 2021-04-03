import datetime

from django.db import models
from django.utils import timezone

class AggData(models.Model):
    pos = models.FloatField()
    neg = models.FloatField()
    neu = models.FloatField()
    comp = models.FloatField()
    brand = models.CharField(max_length=40)
    day = models.DateField()
    ticker = models.CharField(max_length=40)
    closeprice = models.FloatField()
    SMA5 = models.FloatField()
    EMA = models.FloatField()
    fut1 = models.FloatField()
    fut2 = models.FloatField()
    fut3 = models.FloatField()
    fut4 = models.FloatField()
    fut5 = models.FloatField()
    def __str__(self):
        return 'Company: {}, Data: {}, Close Price: {}, Compound Sentiment Score: {}'.format(self.brand, self.day, self.closeprice, self.comp)



# Create your models here.
class Predictions(models.Model):
    p_sentiment = models.FloatField()
    p_5day = models.FloatField()
    p_5dayEMA = models.FloatField()
    p_EMA = models.FloatField()
    p_sentimentEMA = models.FloatField()
    p_realprice = models.FloatField()
    brand = models.CharField(max_length=40)
    def __str__(self):
        return 'Company: {}, Actual Price: {}, Predicted Price: {}'.format(self.brand, self.p_realprice, self.p_sentimentEMA)


class CompanyInfo(models.Model):
    MA5score = models.FloatField()
    EMA5score = models.FloatField()
    EMAscore = models.FloatField()
    sentMAscore = models.FloatField()
    Brand = models.CharField(max_length=40)
    CEO = models.CharField(max_length=40)
    Revenue = models.CharField(max_length=40)
    Industry = models.CharField(max_length=40)
    Location= models.CharField(max_length=40)
    Employees = models.CharField(max_length=40)
    def __str__(self):
        return 'Company: {}, Revenue: {}, Industry: {}, Location: {}'.format(self.Brand, self.Revenue, self.Industry, self.Location)



# PART OF THE DJANGO TUTORIAL ---------------------------------------------------------------------------------------------------------------------------------
class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)

    def __str__(self):
        return self.question_text


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text


class TestQ(models.Model):
    question_text = models.CharField(max_length=200)

    def __str__(self):
        return self.question_text
