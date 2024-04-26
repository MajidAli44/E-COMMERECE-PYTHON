from django.shortcuts import render
from rest_framework import routers, serializers, viewsets, permissions
from .serializers import *
from .models import *


from django.http import JsonResponse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from django.shortcuts import render
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Sum
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes,api_view


from sklearn.decomposition import PCA
from django.db.models import Q
from django.shortcuts import render
from .models import Order, Products
from django.http import JsonResponse,HttpResponse 
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as img_preprocessing
import numpy as np
import requests
from PIL import Image
import io
from Eshop import settings
import requests
from django.http import HttpResponse
from rest_framework import generics
import requests
from django.core.paginator import Paginator
from rest_framework.pagination import PageNumberPagination


# Create your views here.

def IndexPageView(request):
    return render(request, "index.html")

def CartPageView(request):
    return render(request, "cart.html")

def CheckoutPageView(request):
    return render(request, "checkout.html")

class ProductPagination(PageNumberPagination):
    page_size = 20

class ProductVIewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing Product instances.
    """
    queryset = Products.objects.all() 
    pagination_class = ProductPagination        
    
    def get_serializer_class(self):
        if self.action in ('create', 'update', 'partial_update',):
            return ProductWriteSerializer
        return ProductReadSerializer
    
    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        total_pages = response.data['count'] // self.pagination_class.page_size
        if response.data['count'] % self.pagination_class.page_size > 0:
            total_pages += 1
        response.data['currentPage'] = self.paginator.page.number
        response.data['totalPages'] = total_pages
        return response
    
class ProductRetrieve(generics.RetrieveAPIView):
    queryset = Products.objects.all()
    serializer_class = ProductReadSerializer
    template_name = "ProductDetail.html"
    
    def get(self, request, *args, **kwargs):
        pk = self.kwargs.get('pk')
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return render(request, self.template_name, {'data': serializer.data})
    

class OrderCreation(generics.CreateAPIView):
    
    serializer_class = OrderWriteSerializer
    queryset = Order.objects.all()

class UserHistoryCreate(generics.CreateAPIView):
    
    serializer_class = OrderWriteSerializer
    queryset = UserHistory.objects.all()


class CartViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing Cart instances.
    """
    serializer_class = CartSerializer
    queryset = Cart.objects.all()
    

    
class OrderViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing Cart instances.
    """
    serializer_class = OrderWriteSerializer
    queryset = Order.objects.all()
    # def get_serializer_class(self):
    #     if self.action in ('create', 'update', 'partial_update'):
    #         return OrderWriteSerializer
    #     return OrderReadSerializer
    # permission_classes = [permissions.IsAuthenticated]

    # @action(detail=False, methods=['get'])
    # def recommended_product(self, request):    
    #     queryset = self.queryset.select_related('product')\
    #     .values('product','product__title','product__unit_price','product__image')\
    #     .annotate(total_sum=Sum('quantity')).order_by('-total_sum')[:10]
    #     return Response(queryset)



def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0
    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def predict_price_view(request):
    if request.method == 'POST':
        # Load the model
        with open('./model/product_price_predictions_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Get the input features from the request
        brand = request.POST.get('brand')
        category = request.POST.get('category')
        discount = request.POST.get('discount')
        sub_category = request.POST.get('sub_category')
        title = request.POST.get('title')

        # Create a DataFrame of the features
        input_features = {
            'brand': brand,
            'category': category,
            'discount': discount,
            'sub_category': sub_category,
            'title': title
        }
        new_data = pd.DataFrame([input_features])

        # Preprocess input features
        # Convert categorical features to label encoding
        categorical_cols = ['brand', 'category', 'sub_category']
        label_encoders = {col: LabelEncoder() for col in categorical_cols}
        for col in categorical_cols:
            new_data[col] = new_data[col].astype(str)
            new_data[col] = label_encoders[col].fit_transform(new_data[col])  # Use the same label encoder as the one used for training

        # Preprocess the 'title' column for word embedding
        new_data['title'] = new_data['title'].apply(lambda x: x.split())  # Tokenize the 'title' column
        word2vec_model = Word2Vec(new_data['title'], min_count=1)
        title_feature_vectors = [average_word_vectors(title, word2vec_model, word2vec_model.wv.index_to_key, 100) for title in new_data['title']]
        title_feature_vectors = pd.DataFrame(title_feature_vectors)

        # Combine word vectors with other numerical features
        new_data = pd.concat([new_data.drop(columns=['title']), title_feature_vectors], axis=1)

        # Convert all column names to strings
        new_data.columns = new_data.columns.astype(str)

        # Make a prediction
        prediction = model.predict(new_data)

        # Return the prediction
        return JsonResponse({'predicted_price': prediction[0]})  # Changed 'prediction' to 'predicted_price'
    
    else:
        # Render the HTML form
        return render(request, 'priceprediction.html')

# Load the KNN model
knn_model = joblib.load('./model/knn_model.pkl')

# Load the VGG16 base model
base_model = VGG16(include_top=False, input_shape=(256, 256, 3))

def preprocess_image(image):
    # Resize and preprocess the image
    image = image.resize((256, 256))
    image = img_preprocessing.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return image

def extract_features(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Extract features using the VGG16 base model
    features = base_model.predict(preprocessed_image)
    return features

# @api_view(['GET'])
# # @permission_classes((IsAuthenticated, ))
# def recommend_products(request, user_id):
#     if request.method == 'GET':
#         try:
#             order = Order.objects.filter(user=user_id).last()
#         except Order.DoesNotExist:
#             return JsonResponse({'error': 'Order not found'}, status=404)

#         previous_orders = Order.objects.filter(user=user_id).latest()
#         user = request.user
#         print("User is---",user)
#         # Extract features from images of previous orders
#         features = []
#         for previous_order in previous_orders:
#             imageUrl = ''
#             url = previous_order.product.image.name
#             print("Image URL OF ", url)
#             if "http://assets.myntassets.com" in url:
#                 print("image assest found")
#                 imageUrl = url
#             else:
#                 imageUrl = settings.SITE_URL + url
#             image = load_image_from_url(imageUrl)
#             # Extract features from the image using the VGG16 base model
#             image_features = extract_features(image)
#             features.append(image_features)

#         combined_features = np.concatenate(features, axis=0)
#         num_samples = combined_features.shape[0]
#         print("num samples---", num_samples)
#         flattened_features = combined_features.reshape(num_samples, -1)
        
#         num_components = min(313, min(combined_features.shape))
#         pca = PCA(n_components=num_components)
#         reduced_features = pca.fit_transform(flattened_features)
        
#         if reduced_features.shape[1] < 313:
#             padding = np.zeros((num_samples, 313 - reduced_features.shape[1]))
#             reduced_features = np.hstack((reduced_features, padding))
#         print("Befire model prediction")
#         # similar_product_ids = knn_model.predict(reduced_features)
#         num_neighbors = 3
#         distances, indices = knn_model.kneighbors(reduced_features, n_neighbors=num_neighbors)
#         similar_product_ids = indices.flatten()
#         print("ID predicted",similar_product_ids)
#         products = Products.objects.filter(id__in=similar_product_ids)
#         serializer = ProductReadSerializer(products, many=True)
#         print("Before serializer------")
#         return Response(serializer.data)
    
#     elif request.method == 'POST':
#         data = request.POST
#         return JsonResponse({'received_data': data})

#     else:
#         # Handle other HTTP methods
#         return JsonResponse({'error': 'Method not allowed'}, status=405)
       
@api_view(['GET'])
def image_proxy(request):
    image_url = request.GET.get('url', '')
    # Fetch the image from the HTTP server using requests library
    response = requests.get(image_url, stream=True)

    if response.status_code == 200:
        # Set the content type based on the response headers
        content_type = response.headers.get('content-type', 'image/jpeg')
        
        # Return the image content with appropriate content type
        return HttpResponse(response.content, content_type=content_type)
    else:
        # If the request failed, return an empty response or handle the error as needed
        return HttpResponse(status=response.status_code)




import pickle
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Products

# Load your model and encoders 
# loaded_model = pickle.load(open('./model/knn_model_features_base.pkl', 'rb'))
# loaded_encoders = pickle.load(open('./model/encoders_features_base.pkl', 'rb'))

# features = ['gender', 'masterCategory', 'subCategory', 'articleType', 'season']

# def Recommend_product(request,user_id):
#     previous_orders = Order.objects.filter(user=user_id).last()
#     print("Last order----", previous_orders.product.id)
#     print("Last order----", previous_orders.product.gender)
#     print("Last order----", previous_orders.product.title)
#     product = previous_orders.product
#     new_data = {  # Replace with your actual new data point
#         'gender': product.gender,
#         'masterCategory': product.mastercategory,
#         'subCategory': product.subcategory,
#         'articleType': product.articletype,
#         'season': product.season
#     }
#     print("New data---", new_data)
#     # data = request.data
#     new_df = pd.DataFrame([new_data])
#     print("Dataframe---", new_df)
#     for feature in features:
#         new_df[feature] = loaded_encoders[feature].transform(new_df[feature])

#         # Predict and filter IDs
#         distances, indices = loaded_model.kneighbors(new_df, n_neighbors=3)
#         valid_ids = []
#         for i in range(len(indices[0])):
#             neighbor_index = indices[0][i]
#             # Get neighbor ID from the database
#             neighbor_id = Products.objects.all()[neighbor_index].id  

#             # Check if ID is valid
#             if Products.objects.filter(id=neighbor_id).exists():
#                 valid_ids.append(neighbor_id)

#         # Return predicted IDs as JSON response
#     print("valid id's ", valid_ids)
#     return HttpResponse(request,{"predicted_ids":previous_orders})





# views.py
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Order, Products  # Adjusted import statement
import pickle

# Load the model and encoders
with open('./model/knn_model_features_base.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('./model/encoders_features_base.pkl', 'rb') as encoder_file:
    loaded_encoders = pickle.load(encoder_file)

# Define the features
features = ['gender', 'masterCategory', 'subCategory', 'articleType', 'season']

@csrf_exempt
def recommend_product(request, user_id):
    if request.method == 'GET':
        try:
            # Fetch the last order for the user
            previous_order = Order.objects.filter(user=user_id).last()
            product = previous_order.product
            
            # Prepare new data point
            new_data = {
                'gender': product.gender,
                'masterCategory': product.mastercategory,
                'subCategory': product.subcategory,
                'articleType': product.articletype,
                'season': product.season
            }
            
            # Encode new data
            new_df = pd.DataFrame([new_data])
            print("New DF---", new_df)
            for feature in features:
                print("Data type----", type(new_df[feature]))
                print("Data----", new_df[feature])
                new_df[feature] = loaded_encoders[feature].transform(new_df[feature])
            
            # Make predictions
            print("before prediction")
            distances, indices = loaded_model.kneighbors(new_df, n_neighbors=3)
            print("Prediction---", indices)
            # Filter valid IDs
            valid_ids = []
            for i in range(len(indices[0])):
                neighbor_index = indices[0][i]
                neighbor_id = Products.objects.all()[neighbor_index].id  # Adjusted model name
                if Products.objects.filter(id=neighbor_id).exists():
                    valid_ids.append(neighbor_id)
            
            print("valid id's ", valid_ids)
            return JsonResponse({"predicted_ids": valid_ids})
            
        
        except Exception as e:
            return JsonResponse({"error": str(e)})
