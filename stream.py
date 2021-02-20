import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import pickle as pkl

st.title("Resale Price Prediction")
st.header("Fill the details given below: ")

#caching-> keeps the site stay while loading or manipulation large datasets.
@st.cache(allow_output_mutation = True)
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)

    with open('companies.pkl', 'rb') as f:
        companies = pkl.load(f)

    with open('model_transform.pkl', 'rb') as f:
        model_transform = pkl.load(f)

    with open('model_names.pkl', 'rb') as f:
        model_names = pkl.load(f)

    with open('color_transform.pkl', 'rb') as f:
        color_transform = pkl.load(f)

    return model, companies, model_transform, model_names, color_transform

#st.spinner allows us to give the user a progress prompt when loading files or model.
with st.spinner('Loading Files....'):
    model, companies, model_transform, model_names, color_transform = load_model()

#select company
comps = tuple(companies.keys())
model_company = st.selectbox("which car do you own", comps)

#car company models, the company which is selected we have to show only those models
#model_transform is a dict, keys are comp and values are list of models.
#we are selecting thode models, whose company been selected
model_cars = model_transform[model_company]
car_model = []

for i in model_cars:
    car_model.append(str(model_company) + ' ' + i)

my_model = st.selectbox("which model you own", car_model)

#mileage -> the km car is driven
mileage = st.slider("Miles Driven", min_value=500, max_value=100000, step=500, value=20000)

#year of purchase
year = st.number_input("Year of purchase", min_value=1990, max_value=2021, value=2015, step=1)
age = 2021 - int(year)

#color of your car
colors = tuple(color_transform.keys())
car_color = st.selectbox("car of your car", colors)

model_owned = my_model.split(' ')
name = ''
for i in model_owned[1:]:
    name += i

name = name.rstrip()

#encode categorical
model_owned = model_names.index(name)
encode_comp = companies[model_company]
encode_color = color_transform[car_color]

encode_comp = [int(i) for i in encode_comp]
encode_color = [int(i) for i in encode_color]

x = [model_owned, age, mileage]
x += encode_comp
x += encode_color

cols = ['model', 'year', 'mileage', 'Audi', 'Chevrolet', 'Chrysler', 'Dodge',
       'Ford', 'Honda', 'Hyundai', 'Kia', 'Mitsubishi', 'Nissan', 'Tesla',
       'Toyota', 'Volvo', 'Beige', 'Black', 'Blue', 'Brown', 'Burgundy',
       'Gray', 'Green', 'Metallic', 'N/A', 'Orange', 'Red', 'Silver', 'White',
       'other']

X = pd.DataFrame(np.array([x]), columns = cols)

with st.spinner("Predicting..."):
    label = model.predict(X)[0]

st.markdown("**Price of your car is**")
st.write('$', np.floor(np.exp(label)))

st.write()
