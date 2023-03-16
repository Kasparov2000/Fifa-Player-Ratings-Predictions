import streamlit as st
import joblib
import numpy as np

# Set the app title
st.title("Fifa Player Ratings Predictor")

movement_reactions = st.slider('Movement Reactions', 0, 100)
potential = st.slider('Potential', 0, 100)
passing = st.slider('Passing', 0, 100)
wage_eur = st.number_input('Wage (€)', 0, 1_000_000)
value_eur = st.number_input('Value (€)', 0, 300_000_000)
mentality_composure = st.slider('Mentality Composure', 0, 100)
dribbling = st.slider('Dribbling', 0, 100)
attacking_short_passing = st.slider('Attacking Short Passing', 0, 100)
international_reputation = st.slider('International Reputation', 0, 5)
mentality_vision = st.slider('Mentality Reactions', 0, 100)

parameters = [
    mentality_vision,
    potential,
    passing,
    wage_eur,
    value_eur,
    mentality_composure,
    dribbling,
    attacking_short_passing,
    international_reputation,
    mentality_vision
]

if st.button('Predict'):
    model = joblib.load('fifa_prediction_model.joblib')
    X = np.array(parameters)
    if any(X <= 0):
        st.markdown('### Inputs must be greater than 0')
    else:
        st.markdown(f'### Prediction is {model.predict([parameters])[0]}')