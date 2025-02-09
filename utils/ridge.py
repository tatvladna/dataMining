from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import os
from utils import grid_cv
RANDOM_STATE=28112024
import dill


# =========================================  ЗАГРУЗКА ДАННЫХ  =======================================================


base_double_features_min_max_scaler = pd.read_csv("../data/learning/base_double_features_min_max_scaler.csv")
base_double_features_standard_scaler = pd.read_csv("../data/learning/base_double_features_standard_scaler.csv")

base_double_target = pd.read_csv("../data/learning/base_double_target.csv")
lasso_target = pd.read_csv("../data/learning/lasso_target.csv")
pca_double_target = pd.read_csv("../data/learning/pca_double_target.csv")

lasso_features_standard_scaler = pd.read_csv("../data/learning/lasso_features_standard_scaler.csv")
lasso_features_min_max_scaler = pd.read_csv("../data/learning/lasso_features_min_max_scaler.csv")

lasso_alpha_standard_scaler =  pd.read_csv("../data/learning/lasso_alpha_standard_scaler.csv")
lasso_alpha_min_max_scaler = pd.read_csv("../data/learning/lasso_alpha_min_max_scaler.csv")

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

lasso_alpha_min_max_scaler = float(list(lasso_alpha_min_max_scaler.columns)[0])
lasso_alpha_standard_scaler = float(list(lasso_alpha_standard_scaler.columns)[0])

# ====================================== ПОСТРОЕНИЕ РЕГРЕССИОННЫХ МОДЕЛЕЙ ==========================================================
output_folder = "../2-mlr"
model_name = "Ridge"
model = Ridge(max_iter=10000) # используется в 'lsqr', 'saga', игнорируется в auto, svd, lsqr, и cholesky

param_grid = {
    'alpha': np.arange(0.01, 80, 0.01),
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'] 
}


results = []

sizes = [1/3, 1/4, 1/5, 1/10]
list_descriptors = {"MinMaxSc": base_double_features_min_max_scaler, "StdSc": base_double_features_standard_scaler, 
                    "LassoStdSc": lasso_features_standard_scaler, "LassoMinMaxSc": lasso_features_min_max_scaler,
                    "PCAStdSc": pca_double_features_standard_scaler, "PCAMinMaxSc": pca_double_features_min_max_scaler}

transformer = None
target = None

for title, descriptors in list_descriptors.items():
    target = base_double_target 
    
    param_grid = {
    'alpha': np.arange(0.01, 20, 0.01),
    'solver': ['auto', 'svd', 'cholesky', 'lsqr',"saga"]
}

    if title == "LassoStdSc":
        # для Lasso удаляли объекты
        target = lasso_target
        param_grid = {
            'alpha': [lasso_alpha_standard_scaler],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        }
        transformer = lasso_features_standard_scaler_transformer
    elif title == "LassoMinMaxSc":
            # для Lasso удаляли объекты
            target = lasso_target
            param_grid = {
            'alpha': [lasso_alpha_min_max_scaler],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        }
            transformer = lasso_features_min_max_scaler_transformer
    elif title=="MinMaxSc":
         transformer = base_double_features_min_max_scaler_transformer
         target=base_double_target
    elif title=="StdSc":
         transformer = base_double_features_standard_scaler_transformer
         target=base_double_target
    elif title == "PCAStdSc":
        target = pca_double_target
        transformer = pca_double_features_standard_scaler_transformer
    elif title == "PCAMinMaxSc":
        target = pca_double_target
        transformer = pca_double_features_min_max_scaler_transformer


    for size in sizes:                    
        log_r2, log_q2, log_rmse, fit_time, best_params, memory_used, model_size  = grid_cv(model = model,
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
            "model": f"{model_name}{title}_{size:.0%}",
            "size_x_test": f"{int(size*100)}%",
            "r2_train": log_r2,
            "q2_test": log_q2,
            "rmse": log_rmse,
            "mean_time_fit_s_cv": fit_time,
            "mean_ram_fit_mb": memory_used,
            "size_model_pipeline_mb": model_size,
            "grade": log_q2 >= 0.65 and log_rmse <= 0.75,
            "params": best_params
        })

table = pd.DataFrame(results)


os.makedirs(output_folder, exist_ok=True)
txt_path = os.path.join(output_folder, 'ridge.txt')
csv_path = os.path.join(output_folder, 'ridge.csv')

with open(txt_path, 'w', encoding='utf-8') as file:
    file.write(table.to_string(index=False))

table.to_csv(csv_path, index=False, encoding='utf-8')
