from fastapi import FastAPI
import pandas as pd
import pickle
import uvicorn

# -----------------------------------
# Initialize App
# -----------------------------------
app = FastAPI()

# -----------------------------------
# Load Model
# -----------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------------
# Home Route
# -----------------------------------
@app.get("/")
def home():

    return {
        "message": "House Price Prediction API Running"
    }

# -----------------------------------
# Prediction Route
# -----------------------------------
@app.get("/predict")
def predict(
    bedrooms: float,
    bathrooms: float,
    sqft_living: float,
    floors: float,
    sqft_above:float,
    sqft_basement:float,
    yr_built:float,
):

    try:

        input_data = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "floors": floors
        }])

        prediction = model.predict(
            input_data
        )[0]

        prediction = round(
            prediction,
            2
        )

        return {
            "Predicted Price": prediction
        }

    except Exception as e:

        return {
            "Error": str(e)
        }

# -----------------------------------
# Run App
# -----------------------------------
if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )