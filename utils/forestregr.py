from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os
from utils import grid_cv
RANDOM_STATE=28112024
import pickle
import dill

# =========================================  ЗАГРУЗКА ДАННЫХ  =======================================================

# Регуляризация не нужна
base_double_features_min_max_scaler = pd.read_csv("../data/learning/base_double_features_min_max_scaler.csv")
base_double_features_standard_scaler = pd.read_csv("../data/learning/base_double_features_standard_scaler.csv")

# с помощью деревьев решений можно строить важность прризнаков
# попробуем отобрать важные признаки на всех дескрипторах с помощью деревьев
all_double_features_min_max_scaler = pd.read_csv("../data/learning/all_double_features_min_max_scaler.csv")
all_double_features_standard_scaler = pd.read_csv("../data/learning/all_double_features_standard_scaler.csv")

base_double_target = pd.read_csv("../data/learning/base_double_target.csv")
lasso_target = pd.read_csv("../data/learning/lasso_target.csv")
pca_double_target = pd.read_csv("../data/learning/pca_double_target.csv")
all_double_target = pd.read_csv("../data/learning/all_double_target.csv")

lasso_features_standard_scaler = pd.read_csv("../data/learning/lasso_features_standard_scaler.csv")
lasso_features_min_max_scaler = pd.read_csv("../data/learning/lasso_features_min_max_scaler.csv")


pca_double_features_standard_scaler = pd.read_csv("../data/learning/pca_double_features_standard_scaler.csv")
pca_double_features_min_max_scaler = pd.read_csv("../data/learning/pca_double_features_min_max_scaler.csv")

with open("../data/learning/pca_double_features_min_max_scaler_transformer.pkl", "rb") as f:
    pca_double_features_min_max_scaler_transformer = dill.load(f)

with open("../data/learning/pca_double_features_standard_scaler_transformer.pkl", "rb") as f:
    pca_double_features_standard_scaler_transformer = dill.load(f)

with open("../data/learning/lasso_features_standard_scaler_transformer.pkl", "rb") as f:
    lasso_features_standard_scaler_transformer = dill.load(f)

with open("../data/learning/lasso_features_min_max_scaler_transformer.pkl", "rb") as f:
    lasso_features_min_max_scaler_transformer = dill.load(f)

with open("../data/learning/base_double_features_min_max_scaler_transformer.pkl", "rb") as f:
    base_double_features_min_max_scaler_transformer = dill.load(f)

with open("../data/learning/base_double_features_standard_scaler_transformer.pkl", "rb") as f:
    base_double_features_standard_scaler_transformer = dill.load(f)



# ================================== ПОСТРОЕНИЕ МОДЕЛИ СЛУЧАЙНОГО ЛЕСА ==================================
output_folder = "../4-trees/forestregr"
model_name = "ForestRegr"
model = RandomForestRegressor()

param_grid = {
    'n_estimators': np.arange(10, 65, 5),
    'max_depth': np.arange(10, 65, 5),
    'bootstrap': [True, False]
}

results = []

# sizes = [1/3, 1/4, 1/5, 1/10] # деревья не так сильно чувствительны к размеру выборки (как линейная регрессия, например)
size = 1/4 
list_descriptors = {"MinMaxSc": base_double_features_min_max_scaler, "StdSc": base_double_features_standard_scaler, 
                    "LassoStdSc": lasso_features_standard_scaler, "LassoMinMaxSc": lasso_features_min_max_scaler,
                    "PCAStdSc": pca_double_features_standard_scaler, "PCAMinMaxSc": pca_double_features_min_max_scaler,
                    "AllStdSc": all_double_features_standard_scaler, "AllMinMaxSc": all_double_features_min_max_scaler}

transformer = None
target = None 

for title, descriptors in list_descriptors.items():

    if title == "LassoStdSc":
        target = lasso_target
        transformer = lasso_features_standard_scaler_transformer
    elif title == "LassoMinMaxSc":
        target = lasso_target
        transformer = lasso_features_min_max_scaler_transformer
    elif title=="MinMaxSc":
         target=base_double_target
         transformer = base_double_features_min_max_scaler_transformer
    elif title=="StdSc":
         target=base_double_target
         transformer = base_double_features_standard_scaler_transformer
    elif title == "PCAStdSc":
        target = pca_double_target
        transformer = pca_double_features_standard_scaler_transformer
    elif title == "PCAMinMaxSc":
        target = pca_double_target
        transformer = pca_double_features_min_max_scaler_transformer
    elif title == "AllStdSc":
        target = all_double_target
        transformer = all_double_features_standard_scaler
    elif title=="AllMinMaxSc":
        target = all_double_target
        transformer = all_double_features_min_max_scaler
                    
    log_r2, log_q2, log_rmse, fit_time, best_params, memory_used, model_size  = grid_cv(model=model,
                                                                                                        model_name=model_name,
                                                                                                        title = title,
                                                                                                        scoring='r2',
                                                                                                        param_grid = param_grid,
                                                                                                        descriptors= descriptors, 
                                                                                                        target= target, 
                                                                                                        size=size,
                                                                                                        output_folder=output_folder, 
                                                                                                        state=RANDOM_STATE,
                                                                                                        task = 'regression',
                                                                                                        transformer=transformer)

    results.append({
        "model": f"{model_name}{title}",
        "size_x_test": f"{int(size*100)}%",
        "r2_train": log_r2,
        "q2_test": log_q2,
        "rmse": log_rmse,
        "mean_time_fit_s_cv": fit_time,
        "mean_ram_fit_mb": memory_used,
        "size_model_pipeline_mb": model_size,
        "grade": log_q2 >= 0.7 and log_rmse <= 0.7,
        "params": best_params
    })

table= pd.DataFrame(results)


os.makedirs(output_folder, exist_ok=True)
txt_path = os.path.join(output_folder, 'forestregr.txt')
csv_path = os.path.join(output_folder, 'forestregr.csv')

with open(txt_path, 'w', encoding='utf-8') as file:
    file.write(table.to_string(index=False))

table.to_csv(csv_path, index=False, encoding='utf-8')