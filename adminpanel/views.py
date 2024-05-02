from django.core.paginator import Paginator
from django.shortcuts import render
from django.http import HttpResponse
from products.models import *
from transformers import pipeline
from rest_framework.decorators import api_view

# Create your views here.

def AdminPage(request):
    return render(request,'adminDashboard.html')

def ProductPage(request):
    if request.method == 'GET':
        products = Products.objects.all()
        paginator = Paginator(products,8)  
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        print("page object", page_obj)
        return render(request,'products.html', context={"products":page_obj})

def OrderPage(request):
    return render(request,'orders.html')

def UserPage(request):
    if request.method == 'GET':
        products = User.objects.all()
        paginator = Paginator(products,8)  
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        print("page object", page_obj)
        return render(request,'users.html', context={"users":page_obj})

def SentimentAnalysisPage(request):
    return render(request,'usersentimentanaylsis.html')


# Initialize a sentiment-analysis pipeline
nlp = pipeline('sentiment-analysis')


def predict_sentiment(request):
    
    """
    Analyzes sentiment for a reviews and returns results with percentages.
    
    """
    products_ = []
    products = Products.objects.filter(id__in=[1163,21379])
    print("products---", products)
    for product in products:
        reviews = Review.objects.filter(product=product.id)
        print("review---", reviews)
        # Initialize counters for positive and negative reviews
        positive_reviews = 0
        negative_reviews = 0

        # Process each review
        if reviews:
            for review in reviews:
                result = nlp(review.description)[0]
                if result['label'] == 'POSITIVE':
                    positive_reviews += 1
                else:
                    negative_reviews += 1

            # Calculate total number of reviews
            total_reviews = len(reviews)

            # Calculate percentages of positive and negative reviews
            percent_positive = (positive_reviews / total_reviews) * 100
            percent_negative = (negative_reviews / total_reviews) * 100
            
            product_review = {
                "title" : product.title,
                "image" : product.image,
                "positive_review" : positive_reviews,
                "negative_review" : negative_reviews,
                "total_review" : total_reviews,
                "percent_positive" : percent_positive,
                "percent_negative" : percent_negative
            }
            products_.append(product_review)

        # Return the results as a JSON response with percentages
    return render(request,'usersentimentanaylsis.html', context={"products":products_})



