import pickle
import numpy as np

# Load saved objects
model = pickle.load(open("model.pkl", "rb"))
scalar = pickle.load(open("scaler.pkl", "rb"))

def predict_charge(age, bmi, children, sex, smoker, region):

    # Encode sex
    sex_code = 1 if sex == "female" else 0

    # Encode smoker
    smoker_code = 1 if smoker == "yes" else 0

    # One-hot region
    northeast = 1 if region == "northeast" else 0
    northwest = 1 if region == "northwest" else 0
    southeast = 1 if region == "southeast" else 0
    southwest = 1 if region == "southwest" else 0

    # Scale numeric values
    scaled_values = scalar.transform([[age, bmi, children]])

    # Combine inputs
    final_input = np.concatenate(
        (scaled_values[0],
         [smoker_code, sex_code, northeast, northwest, southeast, southwest])
    )

    prediction = model.predict([final_input])[0]
    return round(prediction, 2)
