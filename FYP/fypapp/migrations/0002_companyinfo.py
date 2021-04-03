# Generated by Django 3.1.6 on 2021-03-04 11:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fypapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CompanyInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('MA5score', models.FloatField()),
                ('EMA5score', models.FloatField()),
                ('EMAscore', models.FloatField()),
                ('sentMAscore', models.FloatField()),
                ('Brand', models.CharField(max_length=40)),
                ('CEO', models.CharField(max_length=40)),
                ('Revenue', models.CharField(max_length=40)),
                ('Industry', models.CharField(max_length=40)),
                ('Location', models.CharField(max_length=40)),
                ('Employees', models.CharField(max_length=40)),
            ],
        ),
    ]