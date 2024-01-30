import streamlit as st
import pickle
import numpy as np

# Load the trained model
# model_file = "Pickle_RL_Model.pkl"
# with open(model_file, 'rb') as file:
model = pickle.load(open("Pickle_RL_Model.pkl", 'rb'))

# Function to predict the label
def predict_label(features):
    arr = np.array(features).reshape(1, -1)
    prediction = model.predict(arr)
    return prediction[0]

# Streamlit app
def main():
    st.title("Crop Label Prediction")

    # Input form for user to enter features
    st.sidebar.header("Input Features")
    N = st.sidebar.slider("N", 0, 140, 50)
    P = st.sidebar.slider("P", 5, 145, 53)
    K = st.sidebar.slider("K", 5, 205, 48)
    temperature = st.sidebar.slider("Temperature", 8.8, 43.7, 25.6)
    humidity = st.sidebar.slider("Humidity", 14.3, 100.0, 71.5)
    ph = st.sidebar.slider("pH", 3.5, 9.9, 6.5)
    rainfall = st.sidebar.slider("Rainfall", 20.2, 298.6, 103.5)

    features = [N, P, K, temperature, humidity, ph, rainfall]

    # Display the user input
    st.write("### User Input Features:")
    st.write("N:", N, "  P:", P, "  K:", K, "  Temperature:", temperature, "  Humidity:", humidity, "  pH:", ph,
             "  Rainfall:", rainfall)

    # Predict the label based on user input
    prediction = predict_label(features)

    # Display the predicted label
    st.write("### Predicted Crop Label:")
    st.write(prediction)


if __name__ == "__main__":
    main()
