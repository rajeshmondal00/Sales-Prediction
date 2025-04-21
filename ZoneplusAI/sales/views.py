from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html')

from django.shortcuts import render
from .models import ForecastResult
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_forecast_chart():
    # Your existing forecasting code here
    # Return a DataFrame with results
    pass

def forecast_view(request):
    # Generate or load forecast data
    forecast_df = generate_forecast_chart()
    
    # Convert to base64 for HTML embedding
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    # Save to database
    for _, row in forecast_df.iterrows():
        ForecastResult.objects.update_or_create(
            product_category=row['Product_Category'],
            state=row['State'],
            date=row['Date'],
            defaults={'predicted_units': row['predicted']}
        )
    
    # Prepare context
    top_categories = ForecastResult.objects.values_list(
        'product_category', flat=True
    ).distinct()[:5]
    
    context = {
        'chart': graphic,
        'forecasts': ForecastResult.objects.all()[:100],
        'top_categories': top_categories,
        'last_updated': ForecastResult.objects.latest('created_at').created_at
    }
    
    return render(request, 'forecasts/charts.html', context)