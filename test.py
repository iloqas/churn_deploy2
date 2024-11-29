import requests

# L'URL de l'API Flask
url = 'http://127.0.0.1:5012/predict'

# Les données à envoyer dans la requête POST
data = {
    'Age': 30,
    'Account_Manager': 1,
    'Years': 5,
    'Num_Sites': 2
}

# Envoi de la requête POST à l'API
response = requests.post(url, data=data)

# Vérifier la réponse du serveur
if response.status_code == 200:
    print("Réponse de l'API : ", response.json())  # Afficher la réponse en JSON
else:
    print(f"Erreur {response.status_code}: {response.text}")
