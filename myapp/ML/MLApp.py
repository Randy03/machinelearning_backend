import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import base64
import io
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cluster import AgglomerativeClustering ,KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVC,SVR
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist


class MLApp():
    def __init__(self, dataset,test_size,y_column,x_columns_cont,x_columns_categ,x_column_id,data_separator):
        self.split = test_size
        self.targetcolumn = y_column
        self.data = pd.read_csv(dataset,sep=data_separator)
        self.contcolumns = x_columns_cont
        self.catcolumns = x_columns_categ
        self.idcolumns = x_column_id
        
    
    def convertCategoriesToOneHot(self,data,categories):
        '''Transforma las columnas categoricas del dataset en nuevas columnas con 0 y 1 para que las pueda
        utilizar el modelo'''
        datafinal = data
        for c in categories:
            dummy = pd.get_dummies(data[c],prefix = c)
                #print(dummy)
            datafinal = datafinal[datafinal.columns.values.tolist()].join(dummy)
            datafinal = datafinal.drop(c,axis = 1)
        return datafinal
    
    def findPolinomialFeaturesNames(self,column_names,pol_degree):
        '''Genera los nombres de cada atributo despues de haber modificado el grado del polinomio'''
        features_list = column_names.copy()
        features_dict = {}
        for i in range(0,len(features_list)):
            features_dict[features_list[i]] = i
            features_list[i] = [features_list[i]]
        #print(features_list[0][-1])
        deg = 1
        last_degree = 1
        for x in features_list:
            if(last_degree<len(x)):
                last_degree = len(x)
                deg = deg +1
            if (deg==pol_degree):
                break
            for y in column_names:
                #temp_list = []
                #for item in x:
                #    temp_list.append(item)
                temp_list = x.copy()
                add_validation = False         
                if (features_dict[x[-1]]<=features_dict[y] and  len(temp_list)<deg+1):                                
                    temp_list.append(y)
                    add_validation = True
                if (len(temp_list)==deg+1 and add_validation == True):    
                    features_list.append(temp_list.copy())
        

        feature_list_final = []
        for elm in features_list:
            feature_list_final.append( '*'.join(elm))
        
        return feature_list_final
    
    def isaNumber(self,input):
        try:
            float(input)
            return True
        except:
            return False

    def prepareDataset(self,fillNaCols,replaceValueCols):
        Y =None
        if self.targetcolumn is not None:
            Y = self.data[self.targetcolumn]
            X = self.data.drop(self.targetcolumn,axis = 1)
        else:
            X = self.data
        
        switcher_dict = {
            'MEAN VALUE': lambda x,y: pd.to_numeric(x[x[y].apply(self.isaNumber)][y], errors='coerce').mean(),
            'MEDIAN VALUE': lambda x,y: pd.to_numeric(x[x[y].apply(self.isaNumber)][y], errors='coerce').median(),
            'MODE VALUE': lambda x,y: x[y].mode()[0]
        }
 

        if (fillNaCols is not None):
            for key in fillNaCols:
                try:
                    function_to_apply = switcher_dict[fillNaCols[key]]
                    replace_value = function_to_apply(X,key)
                except KeyError:
                    replace_value = fillNaCols[key]
                X[key] = X[key].fillna(replace_value)
      
        if(replaceValueCols is not None):
            for key in replaceValueCols:
                for value in replaceValueCols[key]:
                    try:
                        function_to_apply = switcher_dict[value[1]]
                        replace_value = function_to_apply(X,key)
                    except KeyError:
                        replace_value = value[1]
                    X[key] = X[key].replace(to_replace=value[0],value=replace_value)                                       

        if len(self.idcolumns)>0:
            X = X.drop(self.idcolumns,axis = 1)
        if len(self.catcolumns)>0:
            X = self.convertCategoriesToOneHot(X,self.catcolumns)
        return X,Y 
    
    def preparePrediction(self,prediction,categories,all_columns):
        '''Transforma las columnas categoricas de la prediccion'''
        x_pred = pd.DataFrame(prediction)
        x_pred = self.convertCategoriesToOneHot(x_pred,categories)
        missing_columns = [x for x in all_columns if x not in x_pred.columns.values.tolist()]
        x_pred = x_pred.join(pd.DataFrame(columns=missing_columns))
        x_pred = x_pred.fillna(0)
        return x_pred

    def load_data(self,polinomial_deg,x_pred=None,fillNaCols=None,replaceValueCols=None):
        X,self.Y = self.prepareDataset(fillNaCols,replaceValueCols)
        X_cols_final = X.columns.values.tolist()
        poly = PolynomialFeatures(degree=polinomial_deg)
        self.X = poly.fit_transform(X)
        self.featureNames = self.findPolinomialFeaturesNames(X_cols_final,polinomial_deg)
        
        if(self.split is not None):
            if (self.Y is not None):
                self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.X,self.Y,test_size = self.split)
            else:
                self.x_train,self.x_test = train_test_split(self.X,test_size = self.split)
        
        self.x_pred = None
        if x_pred is not None:
            x_pred = self.preparePrediction(x_pred,self.catcolumns,X_cols_final)
            self.x_pred = poly.fit_transform(x_pred[X_cols_final])

    
    def linear_regression(self):
        X = self.X
        Y = self.Y
        X_train = self.x_train
        Y_train = self.y_train
        X_test = self.x_test
        Y_test = self.y_test
        x_pred = self.x_pred
        lm = LinearRegression()
        lm.fit(X_train,Y_train)
        intercept = lm.intercept_
        coef = list(zip(self.featureNames,lm.coef_[1:]))
        r_2 = lm.score(X_train,Y_train)
        y_pred =  lm.predict(X_test)
        error_2 = sum((y_pred-Y_test)**2)/len(X_test)
        
        y_pred = None
        if x_pred is not None:
            y_pred = lm.predict(x_pred).tolist()
        result = {
            "R_2": r_2,
            "bias": intercept,
            "featuresCoefficients": coef,
            "prediction": y_pred,
            "testDatasetError": error_2
            
        }
       
        return result

    def logistic_regression(self):
        X = self.X
        Y = self.Y
        X_train = self.x_train
        Y_train = self.y_train
        X_test = self.x_test
        Y_test = self.y_test
        x_pred = self.x_pred
        logm= LogisticRegression()
        logm.fit(X_train,Y_train)
        intercept = logm.intercept_
        #coef = list(zip(X.columns.values.tolist(),logm.coef_))
        #coef = list(zip(self.featureNames,logm.coef_[1:]))
        
        categ = pd.unique(pd.Series(Y))

        coef = {}
        bias = {}
        if(len(categ)==2):
            coef["coef"] = list(zip(self.featureNames,logm.coef_[0,1:]))
            bias["bias"] = intercept.tolist()[0]
        else:
            for i in range(0,len(categ)):
                coef[categ[i]] = list(zip(self.featureNames,logm.coef_[i,1:]))
                bias[categ[i]] = intercept.tolist()[i]
        #coef = logm.coef_
        r_2 = logm.score(X,Y)
        y_pred = logm.predict(X_test)
        test_accuracy = metrics.accuracy_score(Y_test,y_pred)
        y_pred = None
        probs = None
        if x_pred is not None:
            y_pred = logm.predict(x_pred).tolist()      
            probabilities = logm.predict_proba(x_pred).tolist()
            probs = []           
            for i in probabilities:
                tempdict = {}
                for j in range (0,len(categ)):
                    tempdict[categ[j]] = i[j]
                probs.append(tempdict.copy())
        result = {
            "R_2": r_2,
            "bias": bias,
            "featuresCoefficients": coef,
            "prediction": y_pred,
            "probabilityPrediction":probs,
            "test_Accuracy": test_accuracy.tolist()
        }
        #print(result)
        return result

    def hierarchy_cluster(self,n_clus=None,dist_treshold=None):
        X = self.X
        x_pred = self.x_pred
        clus = AgglomerativeClustering(n_clusters=n_clus,distance_threshold=dist_treshold,linkage="ward")
        clus.fit(X)
        labels = clus.labels_.tolist()
        num_leaves = clus.n_leaves_
        
        num_clusters = clus.n_clusters_
        
        plt.figure(figsize=(25,10))
        plt.title("Dendrograma jerarquico truncado")
        plt.xlabel("Indices de la muestra")
        plt.ylabel("Distancias")
        dendrogram(linkage(X,"ward"),leaf_rotation=90.,leaf_font_size=10.0,color_threshold=0.7*180, truncate_mode="lastp",p=10,show_leaf_counts=False, show_contracted=True)
        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read()).decode('utf-8')
        plt.clf()
        result = {
        "labels": labels,
        "leaves": num_leaves,
        "numberOfClusters":num_clusters,
        #"prediction": y_pred,
        "dendogram": pic_hash
        }
        return result

    def KMeans_cluster(self,n_clus_min,n_clus_max):
        X = self.X
        x_pred = self.x_pred
        results = []
        distances = []
        for i in range(n_clus_min,n_clus_max+1):
            clus = KMeans(n_clusters=i)
            clus.fit(X)
            labels = clus.labels_.tolist()
            inertia = clus.inertia_
            centers = {}
            for j in range(0,len(clus.cluster_centers_)):
                centers[j]=list(zip(self.featureNames,clus.cluster_centers_[j,1:]))
            
            y_pred = None
            #if x_pred is not None:
            #    y_pred = clus.predict(X,x_pred)
            distances.append(sum(np.min(cdist(X, clus.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
            result = {
            #"labels": labels,
            "inertia": inertia,
            "centers":centers,
            #"prediction": y_pred,
            }
            results.append(result.copy())
        
        plt.plot(range(n_clus_min,n_clus_max+1), distances, 'bx-')
        plt.xlabel('number of clusters')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method')

        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read()).decode('utf-8')
        plt.clf()
        final_results = {
            "clus_results": results,
            "elbowGraph": pic_hash
        }
        return final_results

    def regression_tree(self,min_samples_split,min_samples_leaf):
        X = self.X[:,1:]
        Y = self.Y
        X_train = self.x_train[:,1:]
        Y_train = self.y_train
        X_test = self.x_test[:,1:]
        Y_test = self.y_test
        x_pred = self.x_pred
        tree = DecisionTreeRegressor(min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,random_state=0)
        tree.fit(X_train,Y_train)
        feature_importance = list(zip(self.featureNames,tree.feature_importances_.tolist()))
        
        
        n_feature = int(tree.n_features_)
        n_classes = int(tree.n_classes_)
        n_outputs = int(tree.n_outputs_)
        score = tree.score(X,Y)
        depth = tree.get_depth()
        n_leaves = int(tree.get_n_leaves())
        
        plt.figure(figsize=(25,10))
        plt.title("Arbol de regression")
        plot_tree(tree,feature_names=self.featureNames)
        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read()).decode('utf-8')
        plt.clf()

        y_pred = tree.predict(X_test)
        error_2 = sum((y_pred-Y_test)**2)/len(X_test)
        y_pred = None

        if x_pred is not None:
            print(x_pred)
            y_pred = tree.predict(x_pred[:,1:]).tolist()
        result = {
        "R_2": score,
        "feature_importance": feature_importance,
        "n_feature":n_feature,
        "n_classes": n_classes,
        "depth":depth,
        "n_leaves":n_leaves,
        "n_outputs":n_outputs,
        "testDatasetError": error_2,
        "y_pred":y_pred,
        "graph":pic_hash
        }
        return result

    def decision_tree(self,min_samples_split,min_samples_leaf):
        X = self.X[:,1:]
        Y = self.Y
        X_train = self.x_train[:,1:]
        Y_train = self.y_train
        X_test = self.x_test[:,1:]
        Y_test = self.y_test
        x_pred = self.x_pred
        tree = DecisionTreeClassifier(criterion="entropy",min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,random_state=99)
        tree.fit(X_train,Y_train)
        feature_importance = list(zip(self.featureNames,tree.feature_importances_.tolist()))
        n_feature = int(tree.n_features_)
        n_classes = int(tree.n_classes_)
        n_outputs = int(tree.n_outputs_)
        score = tree.score(X,Y)
        depth = tree.get_depth()
        n_leaves = int(tree.get_n_leaves())

        plt.figure(figsize=(25,10))
        plt.title("Arbol de decision")
        plot_tree(tree,feature_names=self.featureNames,class_names=pd.unique(pd.Series(Y)).tolist())
        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read()).decode('utf-8')
        plt.clf()

        y_pred = tree.predict(X_test)
        test_accuracy = metrics.accuracy_score(Y_test,y_pred)
        y_pred = None
        if x_pred is not None:
            y_pred = tree.predict(x_pred[:,1:]).tolist()
        result = {
        "R_2": score,
        "feature_importance": feature_importance,
        "n_feature":n_feature,
        "n_classes": n_classes,
        "depth":depth,
        "n_leaves":n_leaves,
        "n_outputs":n_outputs,
        "y_pred":y_pred,
        "test_accuracy":test_accuracy,
        "graph":pic_hash
        }
        return result

    def random_forest_regression(self,number_of_trees,min_samples_split,min_samples_leaf):
        X = self.X[:,1:]
        Y = self.Y
        X_train = self.x_train[:,1:]
        Y_train = self.y_train
        X_test = self.x_test[:,1:]
        Y_test = self.y_test
        x_pred = self.x_pred
        reg_forest = RandomForestRegressor(oob_score=True,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,n_estimators=number_of_trees)
        reg_forest.fit(X_train,Y_train)
        score = reg_forest.score(X,Y)
        feature_importance = list(zip(self.featureNames,reg_forest.feature_importances_.tolist()))
        n_feature = int(reg_forest.n_features_)
        n_outputs = int(reg_forest.n_outputs_)
        oob_score = reg_forest.oob_score_
        oob_predictions = reg_forest.oob_prediction_
        oob_test_error = sum((oob_predictions-Y_train)**2)/len(X)

        y_pred =  reg_forest.predict(X_test)
        error_2 = sum((y_pred-Y_test)**2)/len(X_test)

        y_pred = None
        if x_pred is not None:
            y_pred = reg_forest.predict(x_pred[:,1:]).tolist()

        result = {
            "R_2": score,
            "outOfBag_score":oob_score,
            "feature_importance": feature_importance,
            "n_feature":n_feature,
            "n_outputs":n_outputs,
            "oob_test_error": oob_test_error,
            "testDatasetError": error_2,
            "y_pred":y_pred,
        }

        return result
    
    def random_forest_classification(self,number_of_trees,min_samples_split,min_samples_leaf):
        X = self.X[:,1:]
        Y = self.Y
        X_train = self.x_train[:,1:]
        Y_train = self.y_train
        X_test = self.x_test[:,1:]
        Y_test = self.y_test
        x_pred = self.x_pred
        clas_forest = RandomForestClassifier(oob_score=True,n_estimators=number_of_trees,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
        clas_forest.fit(X_train,Y_train)
        score = clas_forest.score(X,Y)
        oob_score = clas_forest.oob_score_
        feature_importance = list(zip(self.featureNames,clas_forest.feature_importances_.tolist()))
        n_feature = int(clas_forest.n_features_)
        n_outputs = int(clas_forest.n_outputs_)
        
        y_pred = clas_forest.predict(X_test)
        test_accuracy = metrics.accuracy_score(Y_test,y_pred)
        y_pred = None
        if x_pred is not None:
            y_pred = clas_forest.predict(x_pred[:,1:]).tolist()
        result = {
            "R_2": score,
            "outOfBag_score":oob_score,
            "feature_importance": feature_importance,
            "n_feature":n_feature,
            "n_outputs":n_outputs,
            "test_accuracy":test_accuracy,        
            "y_pred":y_pred,
        }
        return result

    def SVM_Regression(self,kernel=None,gamma=None,epsilon=None,C=None,degree=None,coef0=None):
        X = self.X[:,1:]
        Y = self.Y
        X_train = self.x_train[:,1:]
        Y_train = self.y_train
        X_test = self.x_test[:,1:]
        Y_test = self.y_test
        x_pred = self.x_pred

        if(kernel==None):
            kernel = 'rbf'
        if(gamma==None):
            gamma = 'scale'
        if(epsilon==None):
            epsilon = 0.1
        if(C==None):
            C = 1.0
        if(coef0==None):
            coef0=0.0
        if(degree==None):
            degree=1
        svr = SVR(kernel=kernel,degree=degree,coef0=coef0,gamma=gamma,epsilon=epsilon,C=C)

        svr.fit(X_train,Y_train)

        score = svr.score(X,Y)
        svr_coeff = svr.dual_coef_
        intercept = svr.intercept_
        indexs = svr.support_
        support_vectors = svr.support_vectors_

        y_pred = svr.predict(X_test)
        error_2 = sum((y_pred-Y_test)**2)/len(X_test)

        y_pred = None
        if x_pred is not None:
            y_pred = svr.predict(x_pred[:,1:]).tolist()

        result = {
            "R_2": score,
            "y_pred":y_pred,
            "testDatasetError": error_2
        }
        return result

    def SVM_Classification(self,kernel=None,gamma=None,C=None,degree=None,coef0=None):
        X = self.X[:,1:]
        Y = self.Y
        X_train = self.x_train[:,1:]
        Y_train = self.y_train
        X_test = self.x_test[:,1:]
        Y_test = self.y_test
        x_pred = self.x_pred

        if(kernel==None):
            kernel = 'rbf'
        if(gamma==None):
            gamma = 'scale'
        if(C==None):
            C = 1.0
        if(coef0==None):
            coef0=0.0
        if(degree==None):
            degree=1
        svc = SVC(kernel=kernel,degree=degree,coef0=coef0,gamma=gamma,C=C)

        svc.fit(X_train,Y_train)

        score = svc.score(X,Y)
        svc_coeff = svc.dual_coef_
        intercept = svc.intercept_
        indexs = svc.support_
        support_vectors = svc.support_vectors_

        y_pred = svc.predict(X_test)
        test_accuracy = metrics.accuracy_score(Y_test,y_pred)

        y_pred = None
        if x_pred is not None:
            y_pred = svc.predict(x_pred[:,1:]).tolist()

        result = {
            "R_2": score,
            "y_pred":y_pred,
            "testAcuraccy": test_accuracy
        }
        return result     


        #def featuresRanking(model,X,Y):
        #    rfe = RFE(model,cantX)
        #    rfe = rfe.fit(X,Y)
        #    support = rfe.support_
        #    rank = rfe.ranking_

    

