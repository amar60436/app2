#!/usr/bin/env python
# coding: utf-8

# In[17]:


from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('my_model.pkl')  # Update the model filename

# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

# Define a route to handle form submissions and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input from the HTML form
        product_id = float(request.form['Product_ID'])
        source = float(request.form['Source'])
        sales_agent = float(request.form['Sales_Agent'])
        location = float(request.form['Location'])
        delivery_mode = float(request.form['Delivery_Mode'])
        
        # Create an input array for the XGBoost model
        input_data = np.array([[product_id, source, sales_agent, location, delivery_mode]])
        
        # Make a prediction using the model
        predicted_score = model.predict(input_data)[0]
        
        # Render the HTML template with the prediction
        return render_template('index.html', prediction=predicted_score)
    
    except Exception as e:
        return render_template('index.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




