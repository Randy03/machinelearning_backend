# Generated by Django 2.2.7 on 2019-11-20 21:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0018_datasetcolumns'),
    ]

    operations = [
        migrations.RenameField(
            model_name='dataset',
            old_name='y_column_name',
            new_name='y_column_type',
        ),
    ]
