# Generated by Django 2.2.7 on 2019-11-14 21:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0011_mlmodel'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mlmodel',
            name='type',
        ),
        migrations.AddField(
            model_name='mlmodel',
            name='modeltype',
            field=models.CharField(blank=True, max_length=60, null=True),
        ),
    ]
