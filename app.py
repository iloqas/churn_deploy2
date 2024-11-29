from flask import Flask, request, render_template
import joblib
import numpy as np

# Charger le modèle de régression logistique sauvegardé
model = joblib.load('DATA/model.pkl')

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Passer 'None' pour l'initialisation

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Utiliser request.form pour extraire les données du formulaire HTML
        age = float(request.form['Age'])
        account_manager = int(request.form['Account_Manager'])
        years = float(request.form['Years'])
        num_sites = int(request.form['Num_Sites'])

        # Créer un tableau numpy pour les données de prédiction
        features = np.array([[age, account_manager, years, num_sites]])

        # Effectuer la prédiction
        prediction = model.predict(features)

        # Convertir la prédiction en un format compréhensible
        result = "Churn" if prediction[0] > 0.5 else "Non Churn"

        # Renvoyer le résultat et réafficher la page
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Erreur: {str(e)}")

# Fonction pour lancer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5012, debug=True)
