from django.shortcuts import render, redirect
from django.utils import timezone
from .models import CommodityData, BufferStock
from django.http import JsonResponse
from django.utils.timezone import now
import json
import datetime
import joblib
import pandas as pd
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

today = timezone.now().date()

def home(request):
    today = date.today()

    # Fetch all commodities in buffer stock
    commodities = BufferStock.objects.all()

    # Prepare a list to store high price commodities
    high_price_commodities = []

    for commodity in commodities:
        # Fetch the price data for each commodity in different districts
        price_data_list = CommodityData.objects.filter(
            Commodity=commodity.Commodity,
            Date=today
        )

        for price_data in price_data_list:
            if price_data.Price > commodity.threshold:  # Compare against individual threshold
                high_price_commodities.append({
                    'commodity': commodity.Commodity,
                    'state':price_data.State,
                    'price': price_data.Price,
                    'threshold': commodity.threshold
                })

    return render(request, 'newhome2.html', {
        'high_price_commodities': high_price_commodities
    })

def predict(request):
    return render(request, 'newpredict.html')

# def commodity_data_submission(request):
#     if request.method == 'POST':
#         commodity = request.POST.get('commodity')
#         district = request.POST.get('district')
#         season = request.POST.get('season')
#         production = request.POST.get('production')
#         state = request.POST.get('state')
#         production = float(production)

#         # Get the AI predicted price
#         ai_price = getPrice_from_model(commodity, district, season, production, state)

#         # Save the data temporarily in the session
#         request.session['commodity_data'] = {
#             'commodity': commodity,
#             'district': district,
#             'season': season,
#             'production': production,
#             'state': state,
#             'price': ai_price
#         }

#         # Redirect to the price fixing page with the predicted price
#         return render(request, 'price_fix.html', {'price': ai_price})

def price_fix(request):
    if request.method == 'POST':
        action = request.POST.get('action')

        # Retrieve the saved commodity data from the session
        commodity_data = request.session.get('commodity_data')

        if action == 'fix-estimated':
            # Use the AI predicted price
            price = commodity_data['price']
        elif action == 'fix-price':
            # Use the manually entered price
            price = request.POST.get('manual-price')

        # Save the data to the database
        CommodityData.objects.create(
            Commodity=commodity_data['commodity'],
            Season=commodity_data['season'],
            Production=commodity_data['production'],
            State=commodity_data['state'],
            Price=price,
        )

        # Clear the session data after saving
        request.session.pop('commodity_data', None)

        # Redirect to a success page or wherever you want
        return redirect('home')  # replace with your success URL

    return redirect('commodity_data_submission')
def get_price_data(request, commodity):

    # Get the current date
    today = now().date()
    
    # Fetch all matching records
    data = CommodityData.objects.filter(Commodity=commodity, Date=today)
    # Format the response data
    response_data = []
    for record in data:
        response_data.append({
            'state': record.State,
            'price': record.Price
        })
    
    # If no data is found, return an empty list
    if not response_data:
        response_data = []
    
    return JsonResponse(response_data, safe=False)


# def buffer_stock(request):
#     today = date.today()
#     commodities = BufferStock.objects.all()  # Fetch all commodities
    
#     high_price_commodities = []
#     for commodity in commodities:
#         # Fetch the price data for each commodity in different districts
#         price_data_list = CommodityData.objects.filter(
#             Commodity=commodity.Commodity,
#             Date=today
#         )

#         for price_data in price_data_list:
#             if price_data.Price > commodity.threshold:  # Compare against individual threshold
#                 high_price_commodities.append({
#                     'commodity': commodity,
#                     'state' : price_data.State,
#                     'price': price_data.Price,
#                     'stock': commodity.stock,
#                     'threshold': commodity.threshold
#                 })

#     return render(request, 'newbuffer.html', {
#         'commodities': commodities,
#         'high_price_commodities': high_price_commodities
#     })


import pandas as pd

def buffer_stock(request):
    today = date.today()
    commodities = BufferStock.objects.all()  # Fetch all commodities

    # Load the dataset from Excel
    df = pd.read_excel(r"C:\Users\KALYAN\OneDrive\Desktop\september_dataset.xlsx")

    high_price_commodities = []
    for commodity in commodities:
        # Fetch the price data for each commodity in different districts
        price_data_list = CommodityData.objects.filter(
            Commodity=commodity.Commodity,
            Date=today
        )

        for price_data in price_data_list:
            if price_data.Price > commodity.threshold:  # Compare against individual threshold
                # Filter for the state and commodity in the Excel data for the year 2022
                production_data_2022 = df[
                    (df['State'] == price_data.State) & 
                    (df['Commodity'] == commodity.Commodity) & 
                    (df['Year'] == 2022)
                ]

                if not production_data_2022.empty:
                    total_production_2022 = production_data_2022['Production'].sum()
                else:
                    total_production_2022 = 0  # Set to 0 if no data is found for 2022

                high_price_commodities.append({
                    'commodity': commodity.Commodity,
                    'state': price_data.State,
                    'price': price_data.Price,
                    'stock': commodity.stock,
                    'threshold': commodity.threshold,
                    'recommended_production': total_production_2022  # Add this line to pass the production data
                })

    return render(request, 'newbuffer.html', {
        'commodities': commodities,
        'high_price_commodities': high_price_commodities
    })



def update_stock(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        commodity_id = data.get('commodity_id')
        new_stock = data.get('new_stock')
        
        try:
            buffer_stock = BufferStock.objects.get(id=commodity_id)
            buffer_stock.stock = new_stock
            buffer_stock.save()
            return JsonResponse({'success': True, 'commodity_name': buffer_stock.Commodity})
        except BufferStock.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Commodity not found'})

    return JsonResponse({'success': False, 'error': 'Invalid request'})

def graph(request):
    return render(request, 'graph.html')



# xgboost_model = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\xgboost1.pkl')
# lr_model = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\linear1.pkl')
# rf_model = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\random1.pkl')

# state_encoder = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\state_encoder1.pkl')
# district_encoder = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\district_encoder1.pkl')
# commodity_encoder = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\commodity_encoder1.pkl')
# season_encoder = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\season_encoder1.pkl')

# def getPrice_from_model(commodity, district, season, production, state):
#     # Create DataFrame for input data
#     input_df = pd.DataFrame({
#         'State': [state],
#         'District': [district],
#         'Commodity': [commodity],
#         'Season': [season],
#         'Production': [production],
#         'Year': [pd.Timestamp.now().year]  # Use current year or any appropriate value
#     })
    
#     # Apply the same preprocessing as during training
#     try:
#         input_df['State'] = state_encoder.transform(input_df['State'])
#         input_df['District'] = district_encoder.transform(input_df['District'])
#         input_df['Commodity'] = commodity_encoder.transform(input_df['Commodity'])
#         input_df['Season'] = season_encoder.transform(input_df['Season'])
#     except ValueError as e:
#         raise ValueError("Error in encoding: ", e)
    
#     # Predict 'Price_LR' using the Linear Regression model
#     # Note: Ensure that the features passed to lr_model match those used during its training
#     input_df['Price_LR'] = lr_model.predict(input_df[['Year']])  # Adjust as needed based on the features used in lr_model
    
#     # Ensure columns match those used in training
#     required_columns = ['State', 'District', 'Commodity', 'Season', 'Production', 'Year', 'Price_LR']
#     for col in required_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0  # Or some default value

#     # Predict using the XGBoost model
#     xgboost_pred = xgboost_model.predict(input_df)
    
#     # Predict using the Random Forest model
#     rf_pred = rf_model.predict(input_df)
    
#     # Combine predictions
#     final_pred = (xgboost_pred + rf_pred) / 2
    
#     return final_pred[0]

#new code
from django.shortcuts import render
import pandas as pd
import joblib

# Load the saved models and encoders
xgboost_model = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\X.pkl')  # XGBoost model
lr_model = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\l.pkl')       # Linear Regression model
state_encoder = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\state.pkl')
commodity_encoder = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\commodity.pkl')
season_encoder = joblib.load(r'C:\Users\KALYAN\OneDrive\Desktop\pricepredictor\pricepredictor\price\templates\season.pkl')

# Function to predict the price using models
def getPrice_from_model(commodity, season, production, state):
    # Create DataFrame for input data
    input_df = pd.DataFrame({
        'State': [state],
        'Commodity': [commodity],
        'Season': [season],
        'Production': [production],
        'Year': [pd.Timestamp.now().year]  # Use current year for prediction
    })

    # Apply the same preprocessing as during training
    input_df['State'] = state_encoder.transform(input_df['State'])
    input_df['Commodity'] = commodity_encoder.transform(input_df['Commodity'])
    input_df['Season'] = season_encoder.transform(input_df['Season'])

    # Predict 'Price_LR' using the Linear Regression model
    input_df['Price_LR'] = lr_model.predict(input_df[['Year', 'Commodity']])

    # Ensure columns match those used in training
    required_columns = ['State', 'Commodity', 'Season', 'Production', 'Year', 'Price_LR']
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value

    # Predict using the XGBoost model
    xgboost_pred = xgboost_model.predict(input_df)

    # Final predicted price
    return xgboost_pred[0]

# View function for submitting commodity data
# def commodity_data_submission(request):
#     if request.method == 'POST':
#         commodity = request.POST.get('commodity')
#         season = request.POST.get('season')
#         production = request.POST.get('production')
#         state = request.POST.get('state')
#         production = float(production)

#         # Get the AI predicted price
#         ai_price = getPrice_from_model(commodity, season, production, state)

#         # Save the data temporarily in the session
#         request.session['commodity_data'] = {
#             'commodity': commodity,
#             'season': season,
#             'production': production,
#             'state': state,
#             'price': ai_price
#         }

#         # Redirect to the price fixing page with the predicted price
#         return render(request, 'price_fix.html', {'price': ai_price})
import numpy as np
from django.shortcuts import render

def commodity_data_submission(request):
    if request.method == 'POST':
        commodity = request.POST.get('commodity')
        season = request.POST.get('season')
        production = request.POST.get('production')
        state = request.POST.get('state')
        
        # Convert production to float
        try:
            production = float(production)
        except (ValueError, TypeError):
            return render(request, 'error.html', {'message': 'Invalid production value'})

        # Get the AI predicted price
        ai_price = getPrice_from_model(commodity, season, production, state)

        # Ensure ai_price is a Python float (convert from NumPy type if necessary)
        if isinstance(ai_price, np.generic):
            ai_price = ai_price.item()

        # Save the data temporarily in the session
        request.session['commodity_data'] = {
            'commodity': commodity,
            'season': season,
            'production': production,
            'state': state,
            'price': ai_price
        }

        # Redirect to the price fixing page with the predicted price
        return render(request, 'newprice_fix.html', {'price': ai_price})

    # If it's not a POST request, return an error
    return render(request, 'error.html', {'message': 'Invalid request method'})

## new code end


# def show_graph(request):
#     # Load the dataset from Excel
#     df = pd.read_excel(r"C:\Users\KALYAN\OneDrive\Desktop\september_dataset.xlsx")


#     # Get user input from the form
#     state = request.GET.get('state')
#     commodity = request.GET.get('commodity')

#     graph_html = None  # Default: no graph until the form is submitted

#     if state and commodity:
#         # Filter the data based on state and commodity
#         filtered_data = df[(df['State'] == state) & (df['Commodity'] == commodity)]

#         if not filtered_data.empty:
#             # Create the 'Year-Season' column for the x-axis
#             filtered_data['Year-Season'] = filtered_data['Year'].astype(str) + '-' + filtered_data['Season'].astype(str)

#             # Create a line plot with filtered data
#             fig = px.line(filtered_data, x='Year-Season', y='Price', title=f'{commodity} Prices in {state}')
#             fig.update_traces(hovertemplate='Year-Season: %{x}<br>Price: %{y}')
            
#             # Convert the Plotly figure to HTML for embedding in the template
#             graph_html = fig.to_html(full_html=False)
#         else:
#             graph_html = "<p>No data available for the selected State and Commodity.</p>"

#     # Render the template with the graph or message
#     return render(request, 'graphs.html', {'graph_html': graph_html})

#graph code 



from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mpl
from io import BytesIO
import base64
import mplcursors  # For hover tooltips
from matplotlib import style
def graphs_plot(request):
    graph_url = None

    if request.method == 'POST':
        # Get the selected state and commodity from the form
        state = request.POST.get('state')
        commodity = request.POST.get('commodity')

        # Example Python code to process the options
        # Load and process the Excel data
        d = pd.read_excel(r"C:\Users\KALYAN\OneDrive\Desktop\september_dataset.xlsx")

        df = pd.DataFrame(d)

        # Filter the DataFrame based on the selected state and commodity
        indices = df[(df['State'] == state) & (df['Commodity'] == commodity)].index

        if indices.empty:
            return render(request, 'newgraphs.html', {'error': f'No data found for {commodity} in {state}.'})

        if not indices.empty:
            start = int(indices[0])
            end = int(indices[-1])
            xval = []
            yval = []

            while start <= end:
                yval.append(int(df['Price'].iloc[start]))
                xval.append(str(df['Year'].iloc[start]) + "-" + str(df['Season'].iloc[start]))
                start += 1

            # Generate the graph
            fig, ax = mpl.subplots(figsize=(6, 6))
            mpl.style.use('Solarize_Light2')
            ax.grid(True)
            ax.plot(xval, yval, marker="o", mec="r", mfc="b", ls="solid", lw=2, label=f"{commodity} Prices in {state}")

            ax.set_xlabel("Year-Season")
            ax.set_ylabel("Price")
            ax.set_title(f"{commodity} Prices in {state}")
            ax.legend()

            # Rotate x-axis labels for better readability
            ax.set_xticks(range(len(xval)))
            ax.set_xticklabels(xval, rotation=45, ha="right")

            # Tight layout to prevent label overlapping
            fig.tight_layout()

            # Add hover tooltips for x and y values using mplcursors
            '''mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(
                f"Year-Season: {xval[sel.target.index]}\nPrice: {yval[sel.target.index]}"))'''

            # Save the plot to a BytesIO object and convert it to base64 for embedding in HTML
            buf = BytesIO()
            mpl.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            graph_url = 'data:image/png;base64,' + image_base64

    return render(request, 'newgraphs.html', {'graph_url': graph_url})
















#graph code end 