from setuptools import setup

setup(
    name="MLOPS_Crypto",                     # Nom du package installé sur Vertex
    version="0.1",

    # On déclare explicitement les packages
    packages=[
        "MLOPS_Crypto",                      # Package principal (src/)
        "MLOPS_Crypto.utils",                # Sous-package utils/
    ],

    # Mapping des répertoires
    package_dir={
        "MLOPS_Crypto": "src",               # src/ -> MLOPS_Crypto/
        "MLOPS_Crypto.utils": "src/utils",   # src/utils/ -> MLOPS_Crypto/utils/
    },

    include_package_data=True,               # Inclure YAML etc.

    package_data={
        "MLOPS_Crypto": ["*.yaml"],          # configs.yaml, params.yaml
    },

    install_requires=[
        "numpy>=1.22",                       # Requis pour TF/pandas
        "pandas",                            # Traitement des données
        "scikit-learn",                      # Normalisation / ML
        "google-cloud-storage",              # Accès GCS
        "python-json-logger>=2.0.4",         # Module correct (pythonjsonlogger)
        "joblib",                            # Pour scaler.pkl
        "pyyaml"                             # Lecture YAML
    ],
)
