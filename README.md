# House Price Prediction

This is a machine learning project that predicts the price of a house using the Ames housing dataset. I built it to learn how a full ML project works, from cleaning data to putting a model online.

Live app: https://house-price-apppp-biwon2lb9zhrab2muxa9zy.streamlit.app/

## What I did

- Cleaned the data and made new features like house age, years since remodel and garage age
- Built a preprocessing pipeline with scikit-learn (scaling, ordinal encoding and one-hot encoding)
- Tried 11 regression models and compared them
- Tuned the best one (ElasticNet) with GridSearchCV
- Saved the full pipeline with joblib
- Made a Streamlit app where you enter house details and get a predicted price

## Result

ElasticNet gave the best score, with an R2 of about 0.91 on the test data. Tree models like RandomForest and XGBoost overfit, so the simpler linear model worked better here.

## App design

The machine learning part (data cleaning, pipeline, model training and tuning) is my own work. For the look of the app, I used HTML and CSS inside Streamlit, and I took help from AI (GitHub Copilot) to write and improve that styling. The prediction code was not changed by it.

## Tools used

Python, pandas, NumPy, scikit-learn, Streamlit, HTML and CSS, GitHub

## Files

- app.py : the Streamlit app
- sample.csv : sample rows used to fill the form
- house_price_pipeline3.pkl : the trained model pipeline
- requirements.txt : packages needed

## Run it on your computer

pip install -r requirements.txt
streamlit run app.py

## Some problems I faced

- The model predicts log of the price, so I had to convert it back to the real price
- The app showed an error because text columns were read as numbers, which I fixed by checking the column type properly
- After the new design, the values inside the input boxes were not visible, so I changed the text color in the CSS
- Deployment failed at first because the main file name was wrong, so I set it to app.py
- I kept the scikit-learn version the same in requirements.txt so the saved model loads fine
