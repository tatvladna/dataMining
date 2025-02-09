from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
from utils import grid_cv
RANDOM_STATE=15012025
import dill


# ========================================== ЗАГРУЗКА ДАННЫХ =======================================================


# посмотрим как SVM-классификатор сработает на базовых дескрипторах 
# и дескрипторов, отобранных методом PCA
base_int_features_min_max_scaler = pd.read_csv("../data/learning/base_int_features_min_max_scaler.csv")
base_int_features_standard_scaler = pd.read_csv("../data/learning/base_int_features_standard_scaler.csv")

base_int_target = pd.read_csv("../data/learning/base_int_target.csv")


with open("../data/learning/base_int_features_min_max_scaler_transformer.pkl", "rb") as f:
    base_int_features_min_max_scaler_transformer = dill.load(f)

with open("../data/learning/base_int_features_standard_scaler_transformer.pkl", "rb") as f:
    base_int_features_standard_scaler_transformer = dill.load(f)


# ======================================== МОДЕЛИРОВАНИЕ ================================================

output_folder = "../7-svm"
model_name = "SVMClsf"
results = []


model = SVC(class_weight='balanced', probability=True) # обязатльно probability=True,  так как затем нужно будет веротяность для построения ROC-кривой

param_grid = {
    'C': np.arange(0.1, 20, 0.1),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}


size = 1/4 
list_descriptors = {"MinMaxSc": base_int_features_min_max_scaler, "StdSc": base_int_features_standard_scaler}



transformer = None
target = None 

sampling = None

samplings = ["NonSampl", "SMOTE", "UnderSampl", "SMOTETomek"]


for title, descriptors in list_descriptors.items():
    for sampling in samplings:

        if title=="MinMaxSc":
            target=base_int_target
            transformer = base_int_features_min_max_scaler_transformer
        elif title=="StdSc":
            target=base_int_target
            transformer = base_int_features_standard_scaler_transformer

        log_f1, log_balanced_acc, log_auc, fit_time, best_params, model_size, memory_usage, balance_train, balance_test = grid_cv(model=model,
                                                                                                                                        sampling = sampling,
                                                                                                                                        model_name=model_name,
                                                                                                                                        title = title,
                                                                                                                                        scoring="balanced_accuracy",
                                                                                                                                        param_grid = param_grid,
                                                                                                                                        descriptors= descriptors, 
                                                                                                                                        target= target, 
                                                                                                                                        size=size,
                                                                                                                                        output_folder=output_folder, 
                                                                                                                                        state=RANDOM_STATE,
                                                                                                                                        transformer=transformer,
                                                                                                                                        task="classification")

        results.append({
            "model": f"{model_name}{sampling}{title}",
            "size_x_test": f"{int(size*100)}%",
            "balance_activity_train": balance_train,
            "balance_activity_test": balance_test,
            "balanced_accuracy": log_balanced_acc,
            "f1": log_f1,
            "auc": log_auc,
            "mean_time_fit_s_cv": fit_time,
            "mean_ram_fit_mb": memory_usage,
            "size_model_pipeline_mb": model_size,
            "mean_ram_fit_mb": memory_usage,
            "grade": log_f1 >= 0.85 and log_balanced_acc >= 0.85,
            "params": best_params
        })

table= pd.DataFrame(results)


os.makedirs(output_folder, exist_ok=True)
txt_path = os.path.join(output_folder, 'svmclsf.txt')
csv_path = os.path.join(output_folder, 'svmclsf.csv')

with open(txt_path, 'w', encoding='utf-8') as file:
    file.write(table.to_string(index=False))

table.to_csv(csv_path, index=False, encoding='utf-8')
