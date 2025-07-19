import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Generate a synthetic housing dataset
@st.cache_data
def generate_dataset(n=150):
    np.random.seed(1)
    sqft = np.random.normal(1600, 450, n)
    noise = np.random.normal(0, 12000, n)
    price = sqft * 105 + noise
    data = pd.DataFrame({'Area (sqft)': sqft, 'Price ($)': price})
    return data

# Train and return a regression model
@st.cache_data
def get_trained_model(data):
    X = data[['Area (sqft)']]
    y = data['Price ($)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse

# Streamlit App UI
def main():
    st.set_page_config(page_title="House Price Estimator", layout="centered")
    st.title("🏡 House Price Estimator")
    st.write("Estimate the selling price of a house based on its size.")

    # Load and train model
    data = generate_dataset()
    model, r2, rmse = get_trained_model(data)

    # Input
    st.subheader("📥 Input House Details")
    sqft_input = st.slider("Select Area (in square feet):", min_value=500, max_value=4000, value=1600, step=50)

    if st.button("Predict Price"):
        prediction = model.predict([[sqft_input]])[0]
        st.subheader("💰 Predicted Sale Price")
        st.success(f"Estimated Price: **${prediction:,.2f}**")
        
        st.subheader("📈 Size vs Price Visualization")
        fig = px.scatter(data, x='Area (sqft)', y='Price ($)', 
                         title="Distribution of House Prices by Size",
                         opacity=0.6)
        fig.add_scatter(x=[sqft_input], y=[prediction], mode='markers',
                        marker=dict(size=12, color='red'), name='Your Prediction')
        st.plotly_chart(fig)

        st.subheader("📊 Model Performance")
        st.markdown(f"- **R² Score:** {r2:.2f}")
        st.markdown(f"- **Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    
    st.markdown("---")

if __name__ == "__main__":
    main()
