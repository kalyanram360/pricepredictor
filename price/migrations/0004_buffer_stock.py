# Generated by Django 5.1 on 2024-09-01 14:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('price', '0003_buffer'),
    ]

    operations = [
        migrations.AddField(
            model_name='buffer',
            name='stock',
            field=models.FloatField(null=True),
        ),
    ]
