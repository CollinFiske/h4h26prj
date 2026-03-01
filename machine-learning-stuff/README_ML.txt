# Machine Learning Stuff
# ======================
# Place your trained model files and notebooks here.
#
# Suggested structure:
#   machine-learning-stuff/
#   |-- model.py              <-- Wrapper that the backend imports
#   |-- model.pkl             <-- Your exported trained model (joblib/pickle)
#   |-- notebooks/            <-- Jupyter notebooks for training / exploration
#   |   |-- training.ipynb
#   |-- data/                 <-- Training data (add to .gitignore if large)
#   |   |-- fire_data.csv
#
# The backend already has the import path set up.
# Just uncomment the import line in backend/routes/predict.py:
#
#   from model import predict as ml_predict
#
# And call ml_predict(...) instead of the placeholder logic.
