from django.contrib import admin
from .models import Question, Choice, AggData, Predictions, CompanyInfo
# Register your models here.
#admin.site.register(Question)
admin.site.register(AggData)
#'admin.site.register(Choice)
admin.site.register(Predictions)
admin.site.register(CompanyInfo)