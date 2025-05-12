import joblib

models = {
    "MDR_Present": joblib.load("mdr_models/MDR_Present_model.pkl"),
    "Dose_Error": joblib.load("mdr_models/Dose_Error_model.pkl"),
    "Timing_Error": joblib.load("mdr_models/Timing_Error_model.pkl"),
    "Route_Error": joblib.load("mdr_models/Route_Error_model.pkl"),
    "Resistance_Type": joblib.load("mdr_models/Resistance_Type_model.pkl"),
    "Infection_Onset": joblib.load("mdr_models/Infection_Onset_model.pkl"),
    "Antibiotic_Response": joblib.load("mdr_models/Antibiotic_Response_model.pkl"),
    "Resistance_Progression": joblib.load("mdr_models/Resistance_Progression_model.pkl"),
    "RL_Policy_Q_table": joblib.load("mdr_models/RL_Policy_Q_table.pkl"),
    "LMIC_Policy_Simulation": joblib.load("mdr_models/LMIC_Policy_Simulation_model.pkl"),
    "Economic_Impact": joblib.load("mdr_models/Economic_Impact_model.pkl"),
}

def predict(model_name, features):
    return models[model_name].predict(features)