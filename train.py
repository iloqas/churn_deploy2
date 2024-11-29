import pandas as pd
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
 

def read_data():
    df = pd.read_csv('DATA/customer_churn.csv')
    return df



def split_data(df):
    X = df[['Num_Sites', 'Age', 'Account_Manager', 'Years']]  # [['Age', 'Total_Purchase', 'Years','Num_Sites']]  # Remplacer par les noms de vos colonnes de caractéristiques
    y = df['Churn']  # Remplacer par le nom de votre colonne cible
    #X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def predict_from_data(X_train, X_test, y_train, y_test):
    #smote = SMOTE(random_state=42)
    #X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(random_state=42)#sm.Logit(y_train, X_train)  # Create the logistic regression model
    results = model.fit(X_train, y_train)

    #print(results.summary())

    y_pred = results.predict(X_test)

    y_pred_class = (y_pred >= 0.5).astype(int)

    # Calculate recall
    recall = recall_score(y_test, y_pred_class)

    # Get the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    """model_params = {
        'coefficients': coefficients,
        'intercept': intercept
    }

    # Save the dictionary to a .pkl file
    joblib.dump(model_params, 'logistic_regression_coefficients.pkl')"""

    #joblib.dump(model, 'DATA/model.pkl')

    return recall,model

def save_model(model):
    joblib.dump(model, 'DATA/model.pkl')

def read_model():
    model = joblib.load('DATA/model.pkl')
    return model

def predict_from_parameters(new_data, model):
    #new_data_vector = np.array(new_data).reshape(1, -1)
    #linear_predictor = np.dot(new_data_vector, model_params['coefficients'].T) + model_params['intercept']
    predicted_probability = model.predict(new_data)#1 / (1 + np.exp(-linear_predictor))[0][0]
    predicted_class = (predicted_probability >= 0.5).astype(int)

    return predicted_class, recall


df = read_data()
X_train, X_test, y_train, y_test = split_data(df)
recall,params = predict_from_data(X_train, X_test, y_train, y_test)
print('recall', recall)
save_model(params)

model = read_model()
predicted_class= predict_from_parameters(X_test, model)
print(predicted_class[0])

recall = recall_score(y_test, predicted_class[0])

print(recall)

