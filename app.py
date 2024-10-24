import streamlit as st
import pandas as pd
from prophet import Prophet

# Create a sample DataFrame for the Excel file
sample_data = {
    'Month': ['Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 
              'Jun-24', 'Jul-24', 'Aug-24'],
    'Sales Amt': [5319, 9990, 7597, 8485, 5243, 5367, 3168, 5202]
}
sample_df = pd.DataFrame(sample_data)

# Save the sample DataFrame to an Excel file
sample_file_path = 'sample_sales_data.xlsx'
sample_df.to_excel(sample_file_path, index=False)

# Streamlit app layout
st.title("Sales Forecasting App")
st.write("Upload your sales data in the format of the sample file below:")

# Display download link for the sample file
st.markdown(f"[Download Sample Sales Data](./{sample_file_path})", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    
    # Display the DataFrame for debugging
    st.write("Uploaded DataFrame:", df)

    # Check for required columns
    if 'Month' in df.columns and 'Sales Amt' in df.columns:
        # Prepare the data for Prophet
        df['Month'] = pd.to_datetime(df['Month'], format='%b-%y', errors='coerce')
        st.write("DataFrame after date conversion:", df)

        # Rename columns for Prophet
        df.rename(columns={'Month': 'ds', 'Sales Amt': 'y'}, inplace=True)

        # Drop rows with NaT values if date conversion failed
        df = df.dropna()
        st.write("DataFrame after dropping NaT values:", df)

        # Check if there are enough rows to fit the model
        if len(df) < 2:
            st.error("The DataFrame must contain at least 2 valid rows for fitting the model.")
        else:
            # Fit the Prophet model
            model = Prophet()
            model.fit(df)

            # Create a future DataFrame for forecasting
            future = model.make_future_dataframe(periods=3, freq='M')  # Forecasting for next 3 months
            forecast = model.predict(future)

            # Display the results
            st.write("Forecasting Results:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3))

            # Save the forecasting results to a new Excel file for download
            forecast_file = "forecasted_sales_data.xlsx"
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(forecast_file, index=False)
            st.success("Forecasting data is ready for download.")
            st.markdown(f"[Download Forecasting Data](./{forecast_file})", unsafe_allow_html=True)
    else:
        st.error("Uploaded file must contain 'Month' and 'Sales Amt' columns.")
