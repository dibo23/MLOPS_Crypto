from setuptools import setup

setup(
    name="MLOPS_Crypto",                     # Nom du package installé sur Vertex
    version="0.1",

    # On déclare explicitement les packages, car find_packages(ne prenait pas les .py simples)
    packages=[
        "MLOPS_Crypto",                      # Package principal (map sur src/)
        "MLOPS_Crypto.utils",                # Sous-package utils/
    ],

    # On mappe nos dossiers locaux vers les packages Python
    package_dir={
        "MLOPS_Crypto": "src",               # Le dossier src/ devient MLOPS_Crypto/
        "MLOPS_Crypto.utils": "src/utils",   # Le dossier src/utils/ devient MLOPS_Crypto/utils/
    },

    include_package_data=True,               # Inclut les fichiers non-Python (YAML, etc.)

    package_data={
        "MLOPS_Crypto": ["*.yaml"],          # Inclure configs.yaml, params.yaml
    },

    install_requires=[
        "numpy>=1.22",                       # Requis par pandas / sklearn / TensorFlow (pas installé sinon)
        "pandas",                            # Traitement des données
        "scikit-learn",                      # Normalisation / ML
        "google-cloud-storage",              # Accès au bucket GCS
        "python-json-logger>=2.0.4",         # Corrige warning des logs Vertex
        "json-logging>=1.3.0",               # Ajoute le module pythonjsonlogger
        "joblib",                            # Pour scaler.pkl
        "pyyaml"                             # Pour lire params.yaml et configs.yaml
    ],
)
