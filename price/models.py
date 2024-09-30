from django.db import models
from django.utils import timezone
# Create your models here.
class CommodityData(models.Model):
    id = models.AutoField(primary_key=True)
    Date = models.DateField(default=timezone.now)
    Season = models.CharField(max_length=50)
    Commodity = models.CharField(max_length=50)
    State = models.CharField(max_length=50)
    Production = models.FloatField()
    Price = models.FloatField()



class BufferStock(models.Model):
    Commodity = models.CharField(max_length=100)
    stock = models.FloatField(null=True)
    threshold = models.FloatField(null=True)


# class alert(models.Model):
#     id = models.AutoField(primary_key=True)
#     date= models.DateField
#     alert = models.CharField(max_length=50)

# class msp(models.Model):
#     id = models.AutoField(primary_key=True)
#     Crop = models.CharField(max_length=50)
#     msp = models.FloatField
