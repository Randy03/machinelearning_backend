#from .models import Item, User,MLModel,DataSet,EvalInput,LinearRegressionInput
from .models import *
from rest_framework import serializers
from .ML.dataset_functions import getDataFrameValues,getDataFrameColumns,getCategoriesOfColumn


class ItemSerializer(serializers.Serializer):
    #_id = serializers.IntegerField(required=False)
    name = serializers.CharField(max_length=60)
    description = serializers.CharField(max_length=60)
    price = serializers.FloatField()
    dateFrom = serializers.DateTimeField(format="%Y-%m-%d") 
    dateTo = serializers.DateTimeField(format="%Y-%m-%d")

    def create(self, validated_data):
        return Item.objects.create(**validated_data)
    
    def update(self,instance,validated_data):
        instance.name = validated_data.get('name',instance.name)
        instance.description = validated_data.get('description',instance.description)
        instance.price = validated_data.get('price',instance.price)
        instance.dateFrom = validated_data.get('dateFrom',instance.dateFrom)
        instance.dateTo = validated_data.get('dateTo',instance.dateTo)
        instance.save()
        return instance

class ItemSerializerGet(serializers.Serializer):
    _id = serializers.IntegerField(required=False)
    name = serializers.CharField(max_length=60)
    description = serializers.CharField(max_length=60)
    price = serializers.FloatField()
    dateFrom = serializers.DateTimeField(format="%Y-%m-%d") 
    dateTo = serializers.DateTimeField(format="%Y-%m-%d")

class MLModelSerializer(serializers.Serializer):
    _id = serializers.IntegerField(required=False)
    model = serializers.CharField(max_length=60)
    modeltype = serializers.CharField(max_length=60)
    description = serializers.CharField(max_length=60)

class DataSetColumnsSerializer(serializers.Serializer):
    #class Meta:
    #    model = DataSetColumns
    #    fields = ['_id','column_name','column_type']
    _id = serializers.IntegerField(required=False)
    #iddataset= serializers.IntegerField()
    column_name = serializers.CharField(max_length=300)
    column_type = serializers.CharField(max_length=100)

class DataSetSerializer(serializers.Serializer):
    _id = serializers.IntegerField(required=False)
    name = serializers.CharField(max_length=60)
    description = serializers.CharField(max_length=60)
    url = serializers.CharField(max_length=300,required=False)
    sep = serializers.CharField(max_length=5)
    y_column_type = serializers.CharField(max_length=50)
    file = serializers.FileField(required=False)
    columns = DataSetColumnsSerializer(many=True, required=False)
    


    def create(self, validated_data):
        
        ds = DataSet.objects.create(**validated_data)
        try:
            columns_data = validated_data.pop('columns')
            for column in columns_data:
                DataSetColumns.objects.create(iddataset=ds,**column)
        except:
            if (ds.file is not None):
                datasetPath = ds.file.path
            else:
                datasetPath = ds.url 
            columns = getDataFrameColumns(datasetPath,ds.sep)
            for c in columns:
                DataSetColumns.objects.create(iddataset=ds,column_name=c)
            return ds
        return ds
    
    def update(self,instance,validated_data):
        instance.name = validated_data.get('name',instance.name)
        instance.description = validated_data.get('description',instance.description)
        instance.url = validated_data.get('url',instance.url)
        instance.sep = validated_data.get('sep',instance.sep)
        instance.y_column_type = validated_data.get('y_column_type',instance.y_column_type)
        instance.file = validated_data.get('file',instance.file)
        
        columns_data = validated_data.pop('columns')
        for column in columns_data:
            try:
                col = DataSetColumns.objects.get(_id=column.pop('_id'))
                col.column_type = column.pop('column_type')
                col.save()
            except KeyError:
                DataSetColumns.objects.create(iddataset=instance,**column)
        
        instance.save()
        return instance


class DataSetInfoSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=60)
    description = serializers.CharField(max_length=60)
    x_cols = serializers.JSONField()
    y_col = serializers.CharField(max_length=100)
    length = serializers.IntegerField()
    values = serializers.JSONField(required=False)



class LinearRegressionSerializer(serializers.Serializer):
    #idmodel = serializers.IntegerField()
    iddataset = serializers.IntegerField()
    data_test_size =serializers.FloatField()
    polinomial_degree = serializers.IntegerField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return LinearRegressionInput(**validated_data)

class LogisticRegressionSerializer(serializers.Serializer):
    #idmodel = serializers.IntegerField()
    iddataset = serializers.IntegerField()
    data_test_size =serializers.FloatField()
    polinomial_degree = serializers.IntegerField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return LogisticRegressionInput(**validated_data)

class HierarchyClusterSerializer(serializers.Serializer):
    #idmodel = serializers.IntegerField()
    iddataset = serializers.IntegerField()
    #polinomial_degree = serializers.IntegerField(required=False)
    x_pred = serializers.DictField(required=False)
    n_clus = serializers.IntegerField(required=False)
    dist_threshold = serializers.FloatField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return HierarchyClusterInput(**validated_data)

class KmeansClusterSerializer(serializers.Serializer):
    #idmodel = serializers.IntegerField()
    iddataset = serializers.IntegerField()
    #polinomial_degree = serializers.IntegerField(required=False)
    n_clus_min = serializers.IntegerField()
    n_clus_max = serializers.IntegerField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return KmeansClusterInput(**validated_data)
    
class RegressionTreeSerializer(serializers.Serializer):  
    iddataset = serializers.IntegerField()
    data_test_size =serializers.FloatField()
    min_samples_split = serializers.IntegerField()
    min_samples_leaf = serializers.IntegerField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return RegressionTreeInput(**validated_data)

class DecisionTreeSerializer(serializers.Serializer):  
    iddataset = serializers.IntegerField()
    data_test_size =serializers.FloatField()
    min_samples_split = serializers.IntegerField()
    min_samples_leaf = serializers.IntegerField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return DecisionTreeInput(**validated_data)

class RegressionRandomForestSerializer(serializers.Serializer):  
    iddataset = serializers.IntegerField()
    data_test_size =serializers.FloatField()
    number_of_trees = serializers.IntegerField()
    min_samples_split = serializers.IntegerField()
    min_samples_leaf = serializers.IntegerField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return RegressionRandomForestInput(**validated_data)

class ClassificationRandomForestSerializer(serializers.Serializer):  
    iddataset = serializers.IntegerField()
    data_test_size =serializers.FloatField()
    number_of_trees = serializers.IntegerField()
    min_samples_split = serializers.IntegerField()
    min_samples_leaf = serializers.IntegerField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)

    def create(self, validated_data):
        return ClassificationRandomForestInput(**validated_data)

class SVMRegressionSerializer(serializers.Serializer):
    iddataset = serializers.IntegerField()
    degree = serializers.IntegerField(required=False)
    coef0 = serializers.FloatField(required=False)
    kernel = serializers.CharField(max_length=60,required=False)
    epsilon = serializers.FloatField(required=False)
    gamma = serializers.FloatField(required=False)
    C = serializers.FloatField(required=False)
    data_test_size = serializers.FloatField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)
    
    def create(self, validated_data):
        return SVMRegressionInput(**validated_data)

class SVMClassificationSerializer(serializers.Serializer):
    iddataset = serializers.IntegerField()
    degree = serializers.IntegerField(required=False)
    coef0 = serializers.FloatField(required=False)
    kernel = serializers.CharField(max_length=60,required=False)
    gamma = serializers.FloatField(required=False)
    C = serializers.FloatField(required=False)
    data_test_size = serializers.FloatField()
    x_pred = serializers.DictField(required=False)
    fillna_cols = serializers.DictField(required=False)
    replace_cols = serializers.DictField(required=False)
    
    def create(self, validated_data):
        return SVMClassificationInput(**validated_data)

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ( 'email', 'password')
        extra_kwargs = {'password': {'write_only': True}}
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user
    
    def update(self,instance,validated_data):
        #instance.name = validated_data.get('name',instance.name)
        #instance.password = validated_data.get('password',instance.password)
        #instance.email = validated_data.get('email',instance.email)
        instance.save()
        return instance