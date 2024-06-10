from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pickle import load

app = Flask(__name__)

# Load the trained model
model = load(open("dia.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        FS = int(request.form['FS'])
        FU = request.form['FU']

        # Convert categorical feature
        FU_value = 1 if FU == 'YES' else 0

        # Create dataframe for prediction
        input_features = pd.DataFrame([[FS, FU_value]], columns=['FS', 'FU_YES'])

        # Make prediction
        prediction = model.predict(input_features)

        result = 'Diabetes: YES' if prediction[0] == 'YES' else 'Diabetes: NO'
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)



