from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Q
from rest_framework import viewsets,views,status
from rest_framework_jwt.views import ObtainJSONWebToken
from rest_framework_jwt.views import JSONWebTokenAPIView
from rest_framework_jwt.authentication import BaseJSONWebTokenAuthentication 
from rest_framework.response import Response
#from .serializers import ItemSerializer,ItemSerializerGet,UserSerializer,MLModelSerializer,DataSetSerializer,ModelEvalSerializer,EvaluatedModelSerializer,DataSetInfoSerializer,LinearRegressionSerializer
from .serializers import *
from .models import *
from rest_framework.permissions import IsAuthenticated
from .ML.dataset_functions import getDataFrameValues,getDataFrameColumns,getCategoriesOfColumn
from .ML.MLApp import MLApp

import jwt,json
import os
# Create your views here.

class ItemApiView(views.APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request,id=None):
        if id is None:
            items = Item.objects.all()
            sz = ItemSerializerGet(items,many=True)
        else:
            item = get_object_or_404(Item.objects.all(),pk=id)
            sz = ItemSerializerGet(item)
        #print(items)

        return Response(sz.data)
        #return Response({"items":items})

    
    def post(self, request):
        sz = ItemSerializer(data=request.data)
        if sz.is_valid(raise_exception=True):
            sz.save()
        return Response({"OK"})

    def put(self, request, id):
        item = get_object_or_404(Item.objects.all(),pk=id)
        sz = ItemSerializer(instance=item,data=request.data,partial=True)
        if sz.is_valid(raise_exception=True):
            sz.save()
        return Response({"OK"})

    def delete(self, request, id):
        item = get_object_or_404(Item.objects.all(),pk=id)
        item.delete()
        return Response({"OK"})

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class MLModelViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer

class DataSetViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    queryset = DataSet.objects.all()
    serializer_class = DataSetSerializer

class DataSetColumnApiView(views.APIView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        dscolumn = DataSetColumnsSerializer(data=request.data)

        if dscolumn.is_valid():
            dsetid = request.data.pop('iddataset')
            print(dsetid)
            dataset = DataSet.objects.filter(pk=dsetid)[0]
            print(dataset)
            dscolumn.save(dataset)
            return Response({"OK"})
        else:
            return Response({"error"})


class DatasetApiView(views.APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request,id=None):
        if id is None:
            datasets = DataSet.objects.all()
            sz = DataSetSerializer(datasets,many=True)
        else:
            dataset = get_object_or_404(DataSet.objects.all(),pk=id)
            datasetPath = ""
            if (dataset.file is not None):
                datasetPath = dataset.file.path
            else:
                datasetPath = dataset.url            
            values,length = getDataFrameValues(datasetPath,200)
            #x_col, length = getDataFrameInfo(dataset.url,dataset.y_column_name)
            x_cols_types = [(r.column_name,r.column_type) for r in DataSetColumns.objects.filter(Q(column_type='Continue') | Q(column_type='Category') ,iddataset=dataset._id)]
            x_cols = []
            
            if (len(x_cols_types)>0):
                for x in x_cols_types:
                    if (x[1]=='Category'):
                        x_cols.append([x[0],getCategoriesOfColumn(dataset.url,x[0])])  
                    else:
                        x_cols.append([x[0],[]])  
            else:
                x_cols = getDataFrameColumns(datasetPath)
            y_col = [r.column_name for r in DataSetColumns.objects.filter(column_type='Target',iddataset=dataset._id)]
            sz = DataSetInfoSerializer(DataSetInfo(dataset.name,dataset.description,x_cols,y_col,length,values))
        #print(items)
    
        return Response(sz.data)
        #return Response({"items":items})
    
    def post(self, request):
    
      file_serializer = DataSetSerializer(data=request.data)

      if file_serializer.is_valid():
          file_serializer.save()
          return Response(file_serializer.data, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def patch(self, request, id):
        ds = get_object_or_404(DataSet.objects.all(),pk=id)
        sz = DataSetSerializer(instance=ds,data=request.data,partial=True)
        print(request.data)
        if sz.is_valid(raise_exception=True):
            sz.save()
        return Response(sz.data)


class MachinLearningApiView(views.APIView):
    def load_dataset(self,input):
        if input.is_valid(raise_exception=True):
            self.input = input.save()
        self.dataset  = get_object_or_404(DataSet.objects.all(),pk=self.input.iddataset)
        self.x_cont_cols = [r.column_name for r in DataSetColumns.objects.filter(column_type='Continue',iddataset=self.input.iddataset)]
        self.x_id_cols = [r.column_name for r in DataSetColumns.objects.filter(column_type='ID',iddataset=self.input.iddataset)]
        self.x_cat_cols = [r.column_name for r in DataSetColumns.objects.filter(column_type='Category',iddataset=self.input.iddataset)]
        self.y_col = [r.column_name for r in DataSetColumns.objects.filter(column_type='Target',iddataset=self.input.iddataset)]

        if (self.dataset.file is not None):
            self.datasetPath = self.dataset.file.path
        else:
            self.datasetPath = self.dataset.url

class LinearRegressionApiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = LinearRegressionSerializer(data=request.data)
        self.load_dataset(input)
        lr = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        lr.load_data(polinomial_deg=self.input.polinomial_degree,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = lr.linear_regression()
        
        return JsonResponse(result)

class LogisticRegressionApiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = LogisticRegressionSerializer(data=request.data)
        self.load_dataset(input)
        lr = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        lr.load_data(polinomial_deg=self.input.polinomial_degree,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = lr.logistic_regression()
        return JsonResponse(result)

                
class HierarchyClusterApiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = HierarchyClusterSerializer(data=request.data)
        self.load_dataset(input)
        hc = MLApp(self.datasetPath,None,None,self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        hc.load_data(polinomial_deg=1,x_pred=self.input.x_pred[0],fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = hc.hierarchy_cluster(n_clus=self.input.n_clus[0],dist_treshold=self.input.dist_threshold)
        return JsonResponse(result)

class KmeansClusterApiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = KmeansClusterSerializer(data=request.data)
        self.load_dataset(input)
        kmc = MLApp(self.datasetPath,None,None,self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        kmc.load_data(polinomial_deg=1,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = kmc.KMeans_cluster(self.input.n_clus_min,self.input.n_clus_max)
        return JsonResponse(result)

class RegressionTreeApiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = RegressionTreeSerializer(data=request.data)
        self.load_dataset(input)
        rt = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        rt.load_data(polinomial_deg=1,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = rt.regression_tree(min_samples_split=self.input.min_samples_split,min_samples_leaf=self.input.min_samples_leaf)
        return JsonResponse(result)

class DecisionTreeAPiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = DecisionTreeSerializer(data=request.data)
        self.load_dataset(input)
        dt = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        dt.load_data(polinomial_deg=1,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = dt.decision_tree(min_samples_split=self.input.min_samples_split,min_samples_leaf=self.input.min_samples_split)
        return JsonResponse(result)

class RegressionRandomForestAPiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = RegressionRandomForestSerializer(data=request.data)
        self.load_dataset(input)
        app = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        app.load_data(polinomial_deg=1,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = app.random_forest_regression(number_of_trees=self.input.number_of_trees,min_samples_split=self.input.min_samples_split,min_samples_leaf=self.input.min_samples_split)
        return JsonResponse(result)

class ClassificationRandomForestAPiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = ClassificationRandomForestSerializer(data=request.data)
        self.load_dataset(input)
        app = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        app.load_data(polinomial_deg=1,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = app.random_forest_classification(number_of_trees=self.input.number_of_trees,min_samples_split=self.input.min_samples_split,min_samples_leaf=self.input.min_samples_split)
        return JsonResponse(result)

class SVMRegressionAPiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = SVMRegressionSerializer(data=request.data)
        self.load_dataset(input)       
        app = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        app.load_data(polinomial_deg=1,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = app.SVM_Regression(degree=self.input.degree,coef0=self.input.coef0,kernel=self.input.kernel,gamma=self.input.gamma,epsilon=self.input.epsilon,C=self.input.C)
        return JsonResponse(result)

class SVMClassificationAPiView(MachinLearningApiView):
    permission_classes = [IsAuthenticated]
    def post(self,request):
        input = SVMClassificationSerializer(data=request.data)
        self.load_dataset(input)       
        app = MLApp(self.datasetPath,self.input.data_test_size,self.y_col[0],self.x_cont_cols, self.x_cat_cols,self.x_id_cols,self.dataset.sep)
        app.load_data(polinomial_deg=1,x_pred=self.input.x_pred,fillNaCols=self.input.fillna_cols,replaceValueCols=self.input.replace_cols)
        result = app.SVM_Classification(degree=self.input.degree,coef0=self.input.coef0,kernel=self.input.kernel,gamma=self.input.gamma,C=self.input.C)
        return JsonResponse(result)




