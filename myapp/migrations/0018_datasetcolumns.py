# Generated by Django 2.2.7 on 2019-11-18 17:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0017_delete_datasetcolumns'),
    ]

    operations = [
        migrations.CreateModel(
            name='DataSetColumns',
            fields=[
                ('_id', models.AutoField(primary_key=True, serialize=False)),
                ('column_name', models.CharField(max_length=300)),
                ('column_type', models.CharField(max_length=100)),
                ('iddataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='myapp.DataSet')),
            ],
        ),
    ]
