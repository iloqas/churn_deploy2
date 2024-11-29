import pandas as pd
import joblib
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, classification_report

# Étape 1 : Charger le modèle sauvegardé
def read_model():
    model = joblib.load('DATA/model.pkl')
    return model

# Étape 2 : Charger les données
def read_data():
    df = pd.read_csv('DATA/customer_churn.csv')
    return df

# Étape 3 : Diviser les données pour tester
def split_data(df):
    X = df[['Num_Sites', 'Age', 'Account_Manager', 'Years']]  # Colonnes des caractéristiques
    y = df['Churn']  # Colonne cible
    return X, y

# Étape 4 : Tester le modèle
def test_model(X, y, model):
    y_pred = model.predict(X)

    # Calculer les métriques
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    # Afficher les résultats
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

# Script principal
if __name__ == "__main__":
    # Charger le modèle
    model = read_model()

    # Charger les données
    df = read_data()

    # Diviser les données
    X, y = split_data(df)

    # Tester le modèle
    test_model(X, y, model)
