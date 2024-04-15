from django.shortcuts import render
from rest_framework import routers, serializers, viewsets
from .serializers import *
from .models import *

from django.http import JsonResponse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from django.shortcuts import render

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
    queryset = Products.objects.all()
    
    def get_serializer_class(self):
        if self.action in ('create', 'update', 'partial_update',):
            return ProductWriteSerializer
        return ProductReadSerializer
    

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
    serializer_class = OrderSerializer
    queryset = Order.objects.all()

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
