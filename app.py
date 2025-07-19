import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle

app=Flask(__name__)

CORS(app)

with open("HeartDiseaseModel.pkl","rb") as f:
   model=pickle.load(f) 

with open("Sex_le.pkl","rb") as f:
   Sex_le=pickle.load(f) 

with open("ChestPainType_le.pkl","rb") as f:
    ChestPainType_le=pickle.load(f) 

with open("ExerciseAngina_le.pkl","rb") as f:
    ExerciseAngina_le=pickle.load(f) 

with open("ST_Slope_le.pkl","rb") as f:
   ST_Slope_le=pickle.load(f) 

with open("RestingECG_le.pkl","rb") as f:
    RestingECG_le=pickle.load(f) 

@app.route("/")
def home():
    return "you are at home"

@app.route("/predict",methods=['POST'])
def predict():
    try:
        data=request.get_json()

        # Validate input
        required_fields = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG',
                           'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        missing = [f for f in required_fields if f not in data]
        # This is a list comprehension. It goes through each required field and checks whether it exists in the incoming data
        # If a required field is not present, it adds it to the missing list.
        if missing:
            # if missing val then it come to missing list
            return jsonify({'error': f'Missing fields: {missing}'}), 400


        df=pd.DataFrame([data])

        df['Sex']=Sex_le.transform([df["Sex"][0]])
        df['ChestPainType']=ChestPainType_le.transform([df["ChestPainType"][0]])
        df['ExerciseAngina']=ExerciseAngina_le.transform([df["ExerciseAngina"][0]])
        df['ST_Slope']=ST_Slope_le.transform([df["ST_Slope"][0]])
        df['RestingECG']=RestingECG_le.transform([df["RestingECG"][0]])

        feature=df[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG',
        'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
        
        pred_val=model.predict(feature)
        # return array

        if pred_val[0]==1:
            return jsonify({"prediction":"Yes"})
        else:
            return jsonify({'prediction':'No'})
    
    except Exception as e:
        return jsonify({"error": str(e)}),400


    



if __name__=='__main__':
    app.run(debug=True)








