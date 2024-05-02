from rest_framework.pagination import PageNumberPagination
from rest_framework.decorators import api_view
from rest_framework import viewsets, generics
from rest_framework.response import Response
from django.shortcuts import render
from .serializers import *
from .models import *

from django.http import JsonResponse, HttpResponse
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as img_preprocessing

from collections import Counter
import numpy as np
import pandas as pd
import pickle
import requests
import random


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
    
    serializer_class = UserHistorySerializer
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


loaded_model = pickle.load(open('./model/knn_model_features_base.pkl', 'rb'))
loaded_encoders = pickle.load(open('./model/encoders_features_base.pkl', 'rb'))
product_predict = pickle.load(open('./model/Y_train.pkl', 'rb')) 


features = ['gender', 'masterCategory', 'subCategory', 'articleType', 'season','unit_price']




def get_knn_recommendations(product_data, num_recs):
    # Encode features of the product
    product_df = pd.DataFrame([product_data])
    for feature in features:
        product_df[feature] = loaded_encoders[feature].transform(product_df[feature])

    # Get nearest neighbors
    distances, indices = loaded_model.kneighbors(product_df, n_neighbors=num_recs)
    predicted_ids = product_predict.iloc[indices[0]].tolist()
    return predicted_ids

@api_view(['GET'])
def Recommend_product(request,user_id):
    previous_orders = Order.objects.filter(user=user_id).order_by('-id')
    user_history = UserHistory.objects.filter(user=user_id).order_by('-id')
    products_ = []
    if previous_orders:
        for order in previous_orders:
            new_data = { 
            'gender': order.product.gender,
            'masterCategory': order.product.mastercategory,
            'subCategory': order.product.subcategory,
            'articleType': order.product.articletype,
            'season': order.product.season,
            }
            print("Aticle type---", order.product.articletype)
            print("unit type---", type(order.product.unit_price))
            products_.append(new_data)
    if user_history:
        for user in user_history:
            new_data = { 
            'gender': user.product.gender,
            'masterCategory': user.product.mastercategory,
            'subCategory': user.product.subcategory,
            'articleType': user.product.articletype,
            'season': user.product.season
            }
            products_.append(new_data)

    if products_:
        num_orders = len(products_)

        # Determine total number of recommendations based on order count
        if num_orders > 3:
            total_recommendations = 10
        else:
            total_recommendations = num_orders * 3  # 3 recommendations per order

        # Categorize products using masterCategory, subCategory, and articleType
        order_categories = [f"{item['gender']}_{item['masterCategory']}_{item['articleType']}" for item in products_] 

        # Count product occurrences per category
        category_counts = Counter(order_categories)

        # Allocate recommendations proportionally (with a minimum)
        min_recs_per_category = 1  # Ensure at least one recommendation per category
        recommendations = []
        for category, count in category_counts.items():
            num_recs_for_category = max(min_recs_per_category, round(total_recommendations * count / len(products_)))
            # Get 3 recommendations for each product in the category
            for product in products_:
                if f"{product['gender']}_{product['masterCategory']}_{product['articleType']}" == category:
                    category_recommendations = get_knn_recommendations(product, 3)
                    recommendations.extend(category_recommendations)

        random.shuffle(recommendations)
        recommended_product = recommendations[:total_recommendations] 
        products = Products.objects.filter(id__in=recommended_product)
        serializer = ProductReadSerializer(products, many=True)
        return Response(serializer.data)
    return Response({})


class ReviewView(generics.CreateAPIView):
    
    serializer_class = ReviewSerializer
    queryset = Review.objects.all()