# Generated by Django 2.2.7 on 2019-12-22 18:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0019_auto_20191120_1821'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='file',
            field=models.FileField(null=True, upload_to=''),
        ),
    ]