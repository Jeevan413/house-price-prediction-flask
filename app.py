from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "change_this_secret_key"  # koi bhi random string

# Model & data load
model = joblib.load("house_price_model.pkl")
df = pd.read_csv("house_data.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        size_raw = request.form.get("size", "").strip()
        bedrooms_raw = request.form.get("bedrooms", "").strip()
        age_raw = request.form.get("age", "").strip()

        # Empty check
        if not size_raw or not bedrooms_raw or not age_raw:
            flash("Please fill all fields before predicting.", "error")
            return redirect(url_for("home"))

        # Number conversion
        try:
            size = float(size_raw)
            bedrooms = int(bedrooms_raw)
            age = float(age_raw)
        except ValueError:
            flash("Please enter valid numeric values.", "error")
            return redirect(url_for("home"))

        # Basic validation
        if size <= 0:
            flash("Size must be greater than 0.", "error")
            return redirect(url_for("home"))

        if bedrooms <= 0:
            flash("Bedrooms must be at least 1.", "error")
            return redirect(url_for("home"))

        if age < 0:
            flash("Age of house cannot be negative.", "error")
            return redirect(url_for("home"))

        # Prepare for model
        features = np.array([[size, bedrooms, age]])

        # Predict
        predicted_price = model.predict(features)[0]
        predicted_price = int(predicted_price)

        flash(f"Estimated House Price: ₹ {predicted_price:,}", "success")
        return redirect(url_for("home"))

    except Exception as e:
        print("ERROR in /predict:", e)
        flash("Something went wrong on the server. Please try again.", "error")
        return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    # Size vs Price
    sizes = df["Size"].tolist()
    prices = df["Price"].tolist()

    # Average price per bedroom
    avg_price_per_bedroom = df.groupby("Bedrooms")["Price"].mean().reset_index()
    bedrooms = avg_price_per_bedroom["Bedrooms"].tolist()
    avg_prices = avg_price_per_bedroom["Price"].tolist()

    return render_template(
        "dashboard.html",
        sizes=sizes,
        prices=prices,
        bedrooms=bedrooms,
        avg_prices=avg_prices,
    )


if __name__ == "__main__":
    app.run(debug=True)
