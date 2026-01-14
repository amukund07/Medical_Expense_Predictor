# Import to read dataset
import numpy as np
import pandas as pd
# Import to make graphs
import matplotlib.pyplot as plt
# Import for model traning
from sklearn.linear_model import LinearRegression 
# Import for Standardization of the weights
from sklearn import preprocessing   
from sklearn.preprocessing import StandardScaler        
# Import for split train test
from sklearn.model_selection import train_test_split

# Import data set
medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')
medical_df = pd.read_csv('medical.csv')

# Sorting data using 1 and 0
smoker_codes = {'no': 0, 'yes': 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

sex_codes={"male":0,"female":1}
medical_df["sex_code"]= medical_df.sex.map(sex_codes)

# One hot encoding 
enc=preprocessing.OneHotEncoder()
enc.fit(medical_df[["region"]])
one_hot = enc.transform(medical_df[['region']]).toarray()
medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot

def estimates(age,bmi,children,sex_code,smoker_code, northwest, southeast, southwest,p,q,r,s,t,u,v,w,b):
    return age*p+bmi*q+children*r+sex_code*s+smoker_code*t+northwest*u+southeast*v+southwest*w+b


def rmse(targets,predictions):
    return  np.sqrt(np.mean(np.square(targets-predictions)))

# Setting up input 
ages=medical_df.age
bmis=medical_df.bmi
childrens=medical_df.children
sex_column = medical_df.sex_code
smoker_column = medical_df.smoker_code
northwests = medical_df.northwest
southeasts = medical_df.southeast
southwests = medical_df.southwest
# Standardization of the weights
# Fitting on scale
numeric_cols = ['age', 'bmi', 'children']
scalar= StandardScaler()
scalar.fit(medical_df[numeric_cols])
scalar_inputs=scalar.transform(medical_df[numeric_cols])


# Combined with the categorical data
cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values

# Training model on the Standarded weights

inputs=np.concatenate((scalar_inputs,categorical_data),axis=1)
targets = medical_df.charges

model= LinearRegression().fit(inputs,targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)

weights_df = pd.DataFrame({
    'feature': np.append(numeric_cols + cat_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df.sort_values('weight', ascending=False)
# Defining for input to chek the data
def predict_charge(age, bmi, children, sex, smoker, region):
    # 1. Encode sex
    sex_code = 1 if sex == "female" else 0
    
    # 2. Encode smoker
    smoker_code = 1 if smoker == "yes" else 0

    # 3. Region one-hot
    northeast = 1 if region == "northeast" else 0
    northwest = 1 if region == "northwest" else 0
    southeast = 1 if region == "southeast" else 0
    southwest = 1 if region == "southwest" else 0

    # 4. Standardize numeric columns
    scaled_values = scalar.transform([[age, bmi, children]])  # scaler already fitted

    # 5. Combine all inputs
    final_input = np.concatenate(
        (scaled_values[0],
        [smoker_code, sex_code, northeast, northwest, southeast, southwest])
    )

    # 6. Predict
    pred = f'{model.predict([final_input])[0]:.2f}'
    
    return pred

