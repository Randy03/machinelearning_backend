from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
class Item(models.Model):
    _id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=60)
    description = models.CharField(max_length=60)
    price = models.FloatField()
    dateFrom = models.DateTimeField() 
    dateTo = models.DateTimeField()

    def __str__(self):
        return self.name 

class MLModel(models.Model):
    _id = models.AutoField(primary_key=True)
    model = models.CharField(max_length=60)
    modeltype = models.CharField(max_length=60, null=True,blank=True)
    description = models.CharField(max_length=60, null=True,blank=True)
    
    def __str__(self):
        return self.model + " " +self.modeltype

class DataSet(models.Model):
    _id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=60)
    description = models.CharField(max_length=60)
    url = models.CharField(max_length=300,null=True)
    sep = models.CharField(max_length=5)
    y_column_type = models.CharField(max_length=50)
    file = models.FileField(null=True)
    

class DataSetColumns(models.Model):
    _id = models.AutoField(primary_key=True)
    iddataset = models.ForeignKey(DataSet, on_delete=models.CASCADE,related_name='columns')
    column_name = models.CharField(max_length=300)
    column_type = models.CharField(max_length=100)
    

class DataSetInfo():
    def __init__(self,name,desc,x_cols,y_col,length,values=None):
        self.name = name 
        self.description = desc
        self.x_cols = x_cols
        self.y_col = y_col
        self.length = length
        self.values = values
        
class User(AbstractUser):
    username = models.CharField(blank=True, null=True,max_length=60)
    email = models.EmailField(('email address'), unique=True)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return "{}".format(self.email)

class BasicMlInput():
    def __init__(self,iddataset,data_test_size,fillna_cols=None,replace_cols=None,x_pred=None):
        self.iddataset = iddataset
        self.data_test_size = data_test_size
        self.fillna_cols = fillna_cols
        self.replace_cols = replace_cols
        self.x_pred = x_pred

class LinearRegressionInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,polinomial_degree,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        self.polinomial_degree = polinomial_degree
        

class LogisticRegressionInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,polinomial_degree,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        self.polinomial_degree = polinomial_degree

class HierarchyClusterInput():
    def __init__(self,iddataset,polinomial_degree=None,n_clus=None,dist_threshold=None,x_pred=None,fillna_cols=None,replace_cols=None):
        self.iddataset = iddataset
        #self.polinomial_degree = polinomial_degree
        self.x_pred = x_pred,
        self.n_clus = n_clus,
        self.dist_threshold = dist_threshold
        self.fillna_cols = fillna_cols
        self.replace_cols = replace_cols

class KmeansClusterInput():
    def __init__(self,iddataset,n_clus_min,n_clus_max,polinomial_degree=None,x_pred=None,fillna_cols=None,replace_cols=None):
        self.iddataset = iddataset
        #self.polinomial_degree = polinomial_degree
        self.n_clus_min = n_clus_min
        self.n_clus_max = n_clus_max
        self.x_pred = x_pred
        self.fillna_cols = fillna_cols
        self.replace_cols = replace_cols

class RegressionTreeInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,min_samples_split,min_samples_leaf,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        #self.polinomial_degree = polinomial_degree
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        

class DecisionTreeInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,min_samples_split,min_samples_leaf,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        #self.polinomial_degree = polinomial_degree
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

class RegressionRandomForestInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,number_of_trees,min_samples_split,min_samples_leaf,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.number_of_trees = number_of_trees
        

class ClassificationRandomForestInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,number_of_trees,min_samples_split,min_samples_leaf,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.number_of_trees = number_of_trees


class SVMRegressionInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,degree=None,coef0=None,kernel=None,gamma=None,epsilon=None,C=None,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        self.degree = degree
        self.coef0 = coef0
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon
        self.C = C

class SVMClassificationInput(BasicMlInput):
    def __init__(self,iddataset,data_test_size,degree=None,coef0=None,kernel=None,gamma=None,C=None,x_pred=None,fillna_cols=None,replace_cols=None):
        super().__init__(iddataset,data_test_size,fillna_cols,replace_cols,x_pred)
        self.degree = degree
        self.coef0 = coef0
        self.kernel = kernel
        self.gamma = gamma
        self.C = C