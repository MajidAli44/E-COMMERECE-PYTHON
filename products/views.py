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


# Create your views here.

def IndexPageView(request):
    return render(request, "index.html")

def CartPageView(request):
    return render(request, "cart.html")

def CheckoutPageView(request):
    return render(request, "checkout.html")

class ProductVIewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing Product instances.
    """
    queryset = Products.objects.filter(id__lte=1850).order_by('id')[:20]         
    
    def get_serializer_class(self):
        if self.action in ('create', 'update', 'partial_update',):
            return ProductWriteSerializer
        return ProductReadSerializer
    

class OrderCreation(generics.CreateAPIView):
    serializer_class = OrderWriteSerializer
    queryset = Order.objects.all()



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

@api_view(['GET'])
# @permission_classes((IsAuthenticated, ))
def recommend_products(request, user_id):
    if request.method == 'GET':
        try:
            order = Order.objects.filter(user=user_id).last()
        except Order.DoesNotExist:
            return JsonResponse({'error': 'Order not found'}, status=404)

        previous_orders = Order.objects.filter(user=user_id)
        user = request.user
        print("User is---",user)
        # Extract features from images of previous orders
        features = []
        for previous_order in previous_orders:
            imageUrl = ''
            url = previous_order.product.image.name
            print("Image URL OF ", url)
            if "http://assets.myntassets.com" in url:
                print("image assest found")
                imageUrl = url
            else:
                imageUrl = settings.SITE_URL + url
            image = load_image_from_url(imageUrl)
            # Extract features from the image using the VGG16 base model
            image_features = extract_features(image)
            features.append(image_features)

        combined_features = np.concatenate(features, axis=0)
        num_samples = combined_features.shape[0]
        print("num samples---", num_samples)
        flattened_features = combined_features.reshape(num_samples, -1)
        
        num_components = min(313, min(combined_features.shape))
        pca = PCA(n_components=num_components)
        reduced_features = pca.fit_transform(flattened_features)
        
        if reduced_features.shape[1] < 313:
            padding = np.zeros((num_samples, 313 - reduced_features.shape[1]))
            reduced_features = np.hstack((reduced_features, padding))
        print("Befire model prediction")
        # similar_product_ids = knn_model.predict(reduced_features)
        num_neighbors = 3
        distances, indices = knn_model.kneighbors(reduced_features, n_neighbors=num_neighbors)
        similar_product_ids = indices.flatten()
        print("ID predicted",similar_product_ids)
        products = Products.objects.filter(id__in=similar_product_ids)
        serializer = ProductReadSerializer(products, many=True)
        print("Before serializer------")
        return Response(serializer.data)
    
    elif request.method == 'POST':
        data = request.POST
        return JsonResponse({'received_data': data})

    else:
        # Handle other HTTP methods
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    
from django.http import JsonResponse
from rest_framework.decorators import api_view
# from .utils import is_token_valid # Assuming the function is in a utils.py file



from rest_framework.authtoken.models import Token

def is_token_valid(token_key):
    """
    Check if the provided token_key is valid.

    Args:
    token_key (str): The token key to check.

    Returns:
    bool: True if the token is valid, False otherwise.
    """
    try:
        token = Token.objects.get(key=token_key)
        return True
    except Token.DoesNotExist:
        return False
    
@api_view(['GET'])
def check_token(request):
    print("called")
    token_key = request.GET.get('token')
    if is_token_valid(token_key):
        return JsonResponse({'valid': True})
    else:
        return JsonResponse({'valid': False})

@api_view(['GET'])
def image_proxy(request):
    # Get the URL of the image from the request query parameters
    print("Requested data---",request.GET)
    image_url = request.GET.get('url', '')
    print("imag url is----", image_url)
    # Fetch the image from the HTTP server using requests library
    response = requests.get(image_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Set the content type based on the response headers
        content_type = response.headers.get('content-type', 'image/jpeg')
        
        # Return the image content with appropriate content type
        return HttpResponse(response.content, content_type=content_type)
    else:
        # If the request failed, return an empty response or handle the error as needed
        return HttpResponse(status=response.status_code)