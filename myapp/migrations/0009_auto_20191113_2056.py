# Generated by Django 2.2.7 on 2019-11-13 23:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0008_auto_20191113_2054'),
    ]

    operations = [
        migrations.RenameField(
            model_name='item',
            old_name='id',
            new_name='_id',
        ),
    ]
