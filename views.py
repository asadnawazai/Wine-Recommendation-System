import joblib
import pandas as pd
from django.shortcuts import render

# Load the column transformer and the model
column_transformer = joblib.load(r'C:\Users\Dell User\Desktop\Real_Food_Wine-Project\Django_Wine_Recomendation\Django_Wine_Recomendation\column_transformer.pkl')
model = joblib.load(r'C:\Users\Dell User\Desktop\Real_Food_Wine-Project\Django_Wine_Recomendation\Django_Wine_Recomendation\wine_recommendation_model.pkl')

def prediction(request):
    prediction = None

    if request.method == 'POST':
        # Extract data from the form
        dish_name = request.POST.get('dish_name')
        ingredients = request.POST.get('ingredients')
        cooking_style = request.POST.get('cooking_style')
        flavors = request.POST.get('flavors')
        texture = request.POST.get('texture')

        # Prepare input data in the expected format
        input_data = {
            'Dish Name': [dish_name],
            'Ingredients': [ingredients],
            'Cooking Style': [cooking_style],
            'Flavors': [flavors],
            'Texture': [texture]
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Apply the column transformer
        try:
            transformed_input = column_transformer.transform(input_df)
            # Make prediction using the model
            prediction = model.predict(transformed_input)[0]
        except Exception as e:
            print(f"Error while making prediction: {e}")

    return render(request, 'form.html', {'prediction': prediction})
