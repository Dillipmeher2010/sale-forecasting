import streamlit as st
import pandas as pd
from prophet import Prophet

# Function to generate a sample Excel file
def create_sample_excel():
    sample_data = {
        "Month": ["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24"],
        "Sales Amt": [5319, 9990, 7597, 8485, 5243, 5367, 3168, 5202],
    }
    df = pd.DataFrame(sample_data)
    return df

# Create the Streamlit app
st.title("Sales Forecasting App")

# Button to download the sample Excel file
if st.button("Download Sample Excel"):
    sample_df = create_sample_excel()
    sample_file = "sample_sales_data.xlsx"
    sample_df.to_excel(sample_file, index=False)
    st.success(f"Sample file '{sample_file}' is ready for download.")
    st.markdown(f"[Download Sample Excel](./{sample_file})", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Process button
if st.button("Process"):
    if uploaded_file is not None:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        
        # Check for required columns
        if 'Month' in df.columns and 'Sales Amt' in df.columns:
            # Prepare the data for Prophet
            df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
            df.rename(columns={'Month': 'ds', 'Sales Amt': 'y'}, inplace=True)
            
            # Fit the Prophet model
            model = Prophet()
            model.fit(df)

            # Create a future DataFrame for forecasting
            future = model.make_future_dataframe(periods=3, freq='M')  # Forecasting for next 3 months
            forecast = model.predict(future)

            # Display the results
            st.write("Forecasting Results:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3))

            # Option to download the forecasting data
            forecast_file = "forecasted_sales_data.xlsx"
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(forecast_file, index=False)
            st.success(f"Forecasting data is ready for download.")
            st.markdown(f"[Download Forecasting Data](./{forecast_file})", unsafe_allow_html=True)
        else:
            st.error("Uploaded file must contain 'Month' and 'Sales Amt' columns.")
    else:
        st.error("Please upload an Excel file.")
