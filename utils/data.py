from descriptors import (load_molecules,
                        get_base_features, get_lasso_features,
                        get_morgan_features, get_pca_features, get_all_features)

import pandas as pd
import dill
from my_logger import logger
import os

# =========================================  ЗАГРУЗКА ДАННЫХ  =======================================================

# нельзя перезаписывать молекулы, т.к потом на них отбираются другие виды дскрипторов
# и выполняется операция удаления дубликатов
MOLECULES = load_molecules("../data/logBCF.sdf")
DOUBLE_TARGET = [m.GetDoubleProp('logBCF') for m in MOLECULES]
INT_TARGET = [m.GetIntProp('class') for m in MOLECULES]

INT_TARGET = pd.Series(INT_TARGET)
DOUBLE_TARGET = pd.Series(DOUBLE_TARGET)

save_dir = "../data/learning"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ----------------------- базовые дескрипторы, отобранные вручную ------------------------------


base_int_target, base_int_features_min_max_scaler, base_int_features_min_max_scaler_transformer = get_base_features(MOLECULES, target=INT_TARGET, scaler_type="MinMaxScaler")
# второй раз не возвращаем таргет, так как алгоритм отбора дескрипторов одинаковый, а масштабирование происходит после отбора 
# то есть таргет одинаковый, нет смысла его перезаписывать
_,  base_int_features_standard_scaler, base_int_features_standard_scaler_transformer = get_base_features(MOLECULES, target=INT_TARGET, scaler_type="StandardScaler")

base_double_target, base_double_features_min_max_scaler, base_double_features_min_max_scaler_transformer = get_base_features(MOLECULES, target=DOUBLE_TARGET, scaler_type="MinMaxScaler")
_,  base_double_features_standard_scaler, base_double_features_standard_scaler_transformer = get_base_features(MOLECULES, target=DOUBLE_TARGET, scaler_type="StandardScaler")


def save_data(features_scaled, transformer, feature_name, data_name):
    features_scaled.to_csv(f"{save_dir}/{data_name}_{feature_name}.csv", index=False)
    with open(f"{save_dir}/{data_name}_{feature_name}_transformer.pkl", 'wb') as f:
        dill.dump(transformer, f)


save_data(base_int_features_min_max_scaler, base_int_features_min_max_scaler_transformer, 'min_max_scaler', 'base_int_features')
save_data(base_int_features_standard_scaler, base_int_features_standard_scaler_transformer, 'standard_scaler', 'base_int_features')

save_data(base_double_features_min_max_scaler, base_double_features_min_max_scaler_transformer, 'min_max_scaler', 'base_double_features')
save_data(base_double_features_standard_scaler, base_double_features_standard_scaler_transformer, 'standard_scaler', 'base_double_features')


base_int_target.to_csv("../data/learning/base_int_target.csv", index=False)
base_double_target.to_csv("../data/learning/base_double_target.csv", index=False)

logger.info("============== Итоговые размеры БАЗОВЫХ сгенерированных данных =============================")
# базовые дескрипторы
logger.info(f"base_int_target: {len(base_int_target)}")
logger.info(f"base_double_target: {len(base_double_target)}")
logger.info(f"base_int_features_min_max_scaler: {len(base_int_features_min_max_scaler)}")
logger.info(f"base_int_features_standard_scaler: {len(base_int_features_standard_scaler)}")
logger.info(f"base_double_features_min_max_scaler: {len(base_double_features_min_max_scaler)}")
logger.info(f"base_double_features_standard_scaler: {len(base_double_features_standard_scaler)}")
logger.info(f"-----------------------------------------------------------------------------------")


# ============================================ ОТБОР ДЕСКРИПТОРОВ С ПОМОЩЬЮ LASSO ===================================================

# в лассо помещаются непрерывные целевые величины (то есть int не будет)
# для дискретных величин используется логистическая регрессия, например

# таргет будет одинаковый (можно не перезаписывать), так как отбор дескрипторов проходил по одинаковому алгоритму
lasso_alpha_standard_scaler, lasso_target, lasso_features_standard_scaler, lasso_features_standard_scaler_transformer = get_lasso_features(MOLECULES, target=DOUBLE_TARGET, scaler_type="StandardScaler", save_dir="../lasso")
lasso_alpha_min_max_scaler, _, lasso_features_min_max_scaler, lasso_features_min_max_scaler_transformer = get_lasso_features(MOLECULES, target=DOUBLE_TARGET, scaler_type="MinMaxScaler", save_dir="../lasso")

save_data(lasso_features_min_max_scaler, lasso_features_min_max_scaler_transformer, 'min_max_scaler', 'lasso_features')
save_data(lasso_features_standard_scaler, lasso_features_standard_scaler_transformer, 'standard_scaler', 'lasso_features')

lasso_target.to_csv("../data/learning/lasso_target.csv", index=False)

# не забываем сохранять альфу
with open("../data/learning/lasso_alpha_min_max_scaler.csv", "w") as f:
    f.write(f"{lasso_alpha_min_max_scaler}\n")

with open("../data/learning/lasso_alpha_standard_scaler.csv", "w") as f:
    f.write(f"{lasso_alpha_standard_scaler}\n")

logger.info("============== Итоговые размеры и значения: Lasso =============================")
# lasso
# logger.info(f"lasso_int_target: {len(lasso_int_target)}") # для упрощения не стали вводить таргет с целыми числами для lasso
logger.info(f"lasso_target: {len(lasso_target)}")
logger.info(f"lasso_features_min_max_scaler: {len(lasso_features_min_max_scaler)}")
logger.info(f"lasso_features_standard_scaler: {len(lasso_features_standard_scaler)}")
logger.info(f"lasso_alpha_min_max_scaler: {lasso_alpha_min_max_scaler}")
logger.info(f"lasso_alpha_standard_scaler: {lasso_alpha_standard_scaler}")
logger.info(f"-----------------------------------------------------------------------------------")


# ====================================  Отпечатки Моргана =====================================


# размер таргета корректируется после удаления nan и удбликатов у фичей
morgan_size256_int_features, morgan_size256_int_target, morgan_size256_int_features_transformer = get_morgan_features(MOLECULES, target=INT_TARGET, radius=3, fpSize=256)
morgan_size256_double_features, morgan_size256_double_target, morgan_size256_double_features_transformer = get_morgan_features(MOLECULES, target=DOUBLE_TARGET, radius=3, fpSize=256)

# pickle не может сохранять lambda-функции
with open("../data/learning/morgan_size256_int_features_transformer.pkl", "wb") as f:
    dill.dump(morgan_size256_int_features_transformer, f)

with open("../data/learning/morgan_size256_double_features_transformer.pkl", "wb") as f:
    dill.dump(morgan_size256_double_features_transformer, f)

morgan_size256_int_target.to_csv("../data/learning/morgan_size256_int_target.csv", index=False)
morgan_size256_double_target.to_csv("../data/learning/morgan_size256_double_target.csv", index=False)

morgan_size256_int_features.to_csv("../data/learning/morgan_size256_int_features.csv", index=False)
morgan_size256_double_features.to_csv("../data/learning/morgan_size256_double_features.csv", index=False)


logger.info("============== Отпечатки Моргана 256 =============================")
# базовые дескрипторы
logger.info(f"morgan_size256_int_target: {len(morgan_size256_int_target)}")
logger.info(f"morgan_size256_double_target: {len(morgan_size256_double_target)}")
logger.info(f"morgan_size256_int_features: {len(morgan_size256_int_features)}")
logger.info(f"morgan_size256_double_features: {len(morgan_size256_double_features)}")
logger.info(f"-----------------------------------------------------------------------------------")


morgan_size512_int_features, morgan_size512_int_target, morgan_size512_int_features_transformer = get_morgan_features(MOLECULES, target=INT_TARGET, radius=3, fpSize=512)
morgan_size512_double_features, morgan_size512_double_target, morgan_size512_double_features_transformer = get_morgan_features(MOLECULES, target=DOUBLE_TARGET, radius=3, fpSize=512)


morgan_size512_int_target.to_csv("../data/learning/morgan_size512_int_target.csv", index=False)
morgan_size512_double_target.to_csv("../data/learning/morgan_size512_double_target.csv", index=False)

morgan_size512_int_features.to_csv("../data/learning/morgan_size512_int_features.csv", index=False)
morgan_size512_double_features.to_csv("../data/learning/morgan_size512_double_features.csv", index=False)

# pickle не может сохранять lambda-функции
with open("../data/learning/morgan_size512_int_features_transformer.pkl", "wb") as f:
    dill.dump(morgan_size512_int_features_transformer , f)

with open("../data/learning/morgan_size512_double_features_transformer.pkl", "wb") as f:
    dill.dump(morgan_size512_double_features_transformer , f)

logger.info("============== Отпечатки Моргана 512: =============================")
# базовые дескрипторы
logger.info(f"morgan_size512_int_target: {len(morgan_size512_int_target)}")
logger.info(f"morgan_size512_double_target: {len(morgan_size512_double_target)}")
logger.info(f"morgan_size512_int_features: {len(morgan_size512_int_features)}")
logger.info(f"morgan_size512_double_features: {len(morgan_size512_double_features)}")
logger.info(f"-----------------------------------------------------------------------------------")


# ===========================================  ОТБОР ДЕСКРИПТОРОВ С ПОМОЩЬЮ PCA ==================================================

pca_int_target, pca_int_features_min_max_scaler, pca_int_features_min_max_scaler_transformer = get_pca_features(MOLECULES, target=INT_TARGET, n_components=0.95, 
                                                                                    scaler_type="MinMaxScaler",
                                                                                    threshold=0.4,
                                                                                    save_dir="../pca")

# таргеты будут одинаковые исходя из алгоритма
# может быть разное количество компонент, но будет одинаковое кол-во объектов
_, pca_int_features_standard_scaler, pca_int_features_standard_scaler_transformer = get_pca_features(MOLECULES, target=INT_TARGET, n_components=0.95, 
                                                                                    scaler_type="StandardScaler",
                                                                                    threshold=0.4,
                                                                                    save_dir="../pca")


pca_double_target, pca_double_features_min_max_scaler, pca_double_features_min_max_scaler_transformer = get_pca_features(MOLECULES, target=DOUBLE_TARGET, n_components=0.95, 
                                                                                    scaler_type="MinMaxScaler",
                                                                                    threshold=0.4,
                                                                                    save_dir="../pca")

_, pca_double_features_standard_scaler, pca_double_features_standard_scaler_transformer = get_pca_features(MOLECULES, target=DOUBLE_TARGET, n_components=0.95, 
                                                                                    scaler_type="StandardScaler",
                                                                                    threshold=0.4,
                                                                                    save_dir="../pca")


save_data(pca_int_features_min_max_scaler, pca_int_features_min_max_scaler_transformer, 'min_max_scaler', 'pca_int_features')
save_data(pca_int_features_standard_scaler, pca_int_features_standard_scaler_transformer, 'standard_scaler', 'pca_int_features')

save_data(pca_double_features_min_max_scaler, pca_int_features_min_max_scaler_transformer, 'min_max_scaler', 'pca_double_features')
save_data(pca_double_features_standard_scaler, pca_int_features_standard_scaler_transformer, 'standard_scaler', 'pca_double_features')

pca_int_target.to_csv("../data/learning/pca_int_target.csv", index=False)
pca_double_target.to_csv("../data/learning/pca_double_target.csv", index=False)

logger.info("============== Итоговые размеры и значения: PCA =============================")
# pca
logger.info(f"pca_int_target: {len(pca_int_target)}")
logger.info(f"pca_double_target: {len(pca_double_target)}")
logger.info(f"pca_int_features_min_max_scaler: {len(pca_int_features_min_max_scaler)}")
logger.info(f"pca_int_features_standard_scaler: {len(pca_int_features_standard_scaler)}")
logger.info(f"pca_double_features_min_max_scaler: {len(pca_double_features_min_max_scaler)}")
logger.info(f"pca_double_features_standard_scaler: {len(pca_double_features_standard_scaler)}")
logger.info(f"-----------------------------------------------------------------------------------")

# # ==================================== И сохраним еще все дескрипторы ==============================


all_int_target, all_int_features_min_max_scaler, all_int_features_min_max_scaler_transformer = get_all_features(MOLECULES, 
                                                                                                        target=INT_TARGET, 
                                                                                                        scaler_type="MinMaxScaler")
_, all_int_features_standard_scaler, all_int_features_standard_scaler_transformer = get_all_features(MOLECULES, 
                                                                                             target=INT_TARGET,
                                                                                             scaler_type="StandardScaler")


all_double_target, all_double_features_min_max_scaler, all_double_features_min_max_scaler_transformer = get_all_features(MOLECULES, 
                                                                                                        target=DOUBLE_TARGET, 
                                                                                                        scaler_type="MinMaxScaler")

# ТАРГет не перезаписываем, алгоритм отбора дескрипторов одинаковый, а скалирование на это не влияет
_, all_double_features_standard_scaler, all_double_features_standard_scaler_transformer = get_all_features(MOLECULES, 
                                                                                             target=DOUBLE_TARGET,
                                                                                             scaler_type="StandardScaler")



all_int_target.to_csv("../data/learning/all_int_target.csv", index=False)
all_double_target.to_csv("../data/learning/all_double_target.csv", index=False)

save_data(all_int_features_min_max_scaler, all_int_features_min_max_scaler_transformer, 'min_max_scaler', 'all_int_features')
save_data(all_int_features_standard_scaler, all_int_features_standard_scaler_transformer, 'standard_scaler', 'all_int_features')

save_data(all_double_features_min_max_scaler, all_double_features_min_max_scaler_transformer, 'min_max_scaler', 'all_double_features')
save_data(all_double_features_standard_scaler, all_double_features_standard_scaler_transformer, 'standard_scaler', 'all_double_features')

logger.info("============== Размеры после генерации всех дескрипторов =============================")
# базовые дескрипторы
logger.info(f"all_int_target: {len(all_int_target)}")
logger.info(f"all_double_target: {len(all_double_target)}")
logger.info(f"all_int_features_min_max_scaler: {len(all_int_features_min_max_scaler)}")
logger.info(f"all_int_features_standard_scaler: {len(all_int_features_standard_scaler)}")
logger.info(f"all_double_features_min_max_scaler: {len(all_double_features_min_max_scaler)}")
logger.info(f"all_double_features_standard_scaler: {len(all_double_features_standard_scaler)}")
logger.info(f"-----------------------------------------------------------------------------------")