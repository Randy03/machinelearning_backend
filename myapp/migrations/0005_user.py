# Generated by Django 2.2.7 on 2019-11-10 16:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0004_auto_20191110_1141'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=60)),
                ('password', models.CharField(max_length=60)),
                ('email', models.CharField(max_length=60)),
            ],
        ),
    ]