from django.urls import include, path
from rest_framework import routers
from . import views
from rest_framework_jwt.views import obtain_jwt_token
from rest_framework_jwt.views import refresh_jwt_token
from rest_framework_jwt.views import verify_jwt_token

router = routers.DefaultRouter()
#router.register(r'items', views.ItemApiView)
router.register(r'user', views.UserViewSet)
router.register(r'mlmodel', views.MLModelViewSet)
#router.register(r'mlmodel', views.MLModelViewSet)
#router.register(r'datasetcolumn', views.DataSetColumnViewSet)
#router.register(r'dataset', views.DataSetViewSet)


urlpatterns = [
    path('', include(router.urls)),
    path('item/',views.ItemApiView.as_view()),
    path('item/<int:id>', views.ItemApiView.as_view()),
    path('dataset/',views.DatasetApiView.as_view()),
    path('datasetcolumn/',views.DataSetColumnApiView.as_view()),
    path('dataset/<int:id>', views.DatasetApiView.as_view()),
    path('user/login',obtain_jwt_token),
    path('user/checkuser', verify_jwt_token),
    path('linear-regression',views.LinearRegressionApiView.as_view()),
    path('logistic-regression',views.LogisticRegressionApiView.as_view()),
    path('hierarchical-clustering',views.HierarchyClusterApiView.as_view()),
    path('kmeans-clustering',views.KmeansClusterApiView.as_view()),
    path('regression-tree',views.RegressionTreeApiView.as_view()),
    path('decision-tree',views.DecisionTreeAPiView.as_view()),
    path('randomforest-regression',views.RegressionRandomForestAPiView.as_view()),
    path('randomforest-classification',views.ClassificationRandomForestAPiView.as_view()),
    path('svm-regression',views.SVMRegressionAPiView.as_view()),
    path('svm-classification',views.SVMClassificationAPiView.as_view()),
    #path('user/login',views.UserApiView.login()),
    #path('user/<int:id>', views.ItemApiView.as_view()),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]