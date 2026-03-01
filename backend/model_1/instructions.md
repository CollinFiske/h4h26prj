# 1. go to backend root
cd /Users/(lillianle is an example)/Documents/GitHub/h4h26prj/backend

# 2. create virtual environment
python3 -m venv venv

# 3. activate it
source venv/bin/activate

# 4. install all packages needed
pip install pandas numpy scikit-learn xgboost joblib requests

# 5. now run model 1
python model_1/fire_predict_model.py

# 6. then run model 2
python model_2/hazard_score_model.py


You'll know the venv is active when your terminal shows `(venv)` at the start of the line:
Ex:
(venv) lillianle@Lillians-MacBook-Air-2 backend %