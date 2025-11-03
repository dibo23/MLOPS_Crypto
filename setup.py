from setuptools import setup, find_packages

setup(
    name="MLOPS_Crypto",  # Nom de ton projet
    version="0.1",  # Version de ton projet
    packages=find_packages(),  # Trouve automatiquement les packages dans le répertoire src
    install_requires=[
        'pandas',             # Pour le traitement des données
        'scikit-learn',       # Pour la normalisation et les modèles ML
        'google-cloud-storage',  # Pour interagir avec Google Cloud Storage
        'tensorflow',         # TensorFlow est déjà dans l'image, mais tu peux le laisser si besoin
    ],
)