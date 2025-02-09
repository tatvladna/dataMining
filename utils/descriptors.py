import pandas as pd
import os
from pandas import DataFrame
from rdkit.Chem import Descriptors, SDMolSupplier
from sklearn.preprocessing import  FunctionTransformer
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from my_logger import logger
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter


# =========================================  ЗАГРУЗКА ДАННЫХ  =======================================================

def standardize_molecules(mols):
    standardized_mols = []
    for mol in mols:
        if mol is None:
            continue

        try:
            # Сохранение свойств молекулы
            properties = {prop: mol.GetProp(prop) for prop in mol.GetPropNames()}

            # Удаление солей
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)

            # Нейтрализация зарядов
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

            # Приведение к родительскому фрагменту
            parent = rdMolStandardize.FragmentParent(mol)
            mol = parent

            # Приведение к стандартной таутомерной форме
            tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
            mol = tautomer_enumerator.Canonicalize(mol)

            # Удаление стереохимии
            Chem.RemoveStereochemistry(mol)

            # Сброс изотопов
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)

            # Канонизация SMILES
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))

            # Добавление и удаление гидрогенов
            mol = Chem.AddHs(mol)
            mol = Chem.RemoveHs(mol)

            # Исправление металлов
            metal_disconnector = rdMolStandardize.MetalDisconnector()
            mol = metal_disconnector.Disconnect(mol)

            # Восстановление свойств молекулы
            for prop, value in properties.items():
                mol.SetProp(prop, value)
            
            # Добавление стандартизированной молекулы в список
            standardized_mols.append(mol)

        except Exception as e:
            logger.error(f"Ошибка обработки молекулы: {e}")

    return standardized_mols # возвращаем список стандартизированных молекул-объектов



def load_molecules(file_path):
    return [mol for mol in SDMolSupplier(file_path) if mol is not None]

# ===================================   ПОДГОТОВКА БАЗОВЫХ ДЕСКРИПТОРОВ  ===============================================

# создаем словарь из дескриторов структуры
ConstDescriptors = {"heavy_atom_count": Descriptors.HeavyAtomCount,
                    "nhoh_count": Descriptors.NHOHCount,
                    "no_count": Descriptors.NOCount,
                    "num_h_acceptors": Descriptors.NumHAcceptors,
                    "num_h_donors": Descriptors.NumHDonors,
                    "num_heteroatoms": Descriptors.NumHeteroatoms,
                    "num_rotatable_bonds": Descriptors.NumRotatableBonds,
                    "num_valence_electrons": Descriptors.NumValenceElectrons,
                    "num_aromatic_rings": Descriptors.NumAromaticRings,
                    "num_Aliphatic_heterocycles": Descriptors.NumAliphaticHeterocycles,
                    "ring_count": Descriptors.RingCount}

# создаем словарь из физико-химических дескрипторов                            
PhisChemDescriptors = {"full_molecular_weight": Descriptors.MolWt,
                       "log_p": Descriptors.MolLogP,
                       "molecular_refractivity": Descriptors.MolMR,
                       "tspa": Descriptors.TPSA, # топологическая полярная поверхность
                        "balaban_j": Descriptors.BalabanJ,
                       }

# объединяем все дескрипторы в один словарь
descriptors = {}
descriptors.update(ConstDescriptors)
descriptors.update(PhisChemDescriptors)



# функция для генерации дескрипторов из молекул
def mol_dsc_calc(mols): 
    df = DataFrame({k: f(m) for k, f in descriptors.items()} 
                     for m in mols)
    return df

descriptors_names = descriptors.keys()
# функция-обертка, чтобы несколько функций в трансформер поместить
def process_molecules(mols):
    standardized_mols = standardize_molecules(mols) # сначала стандартизируем
    return mol_dsc_calc(standardized_mols)

# оформляем sklearn трансформер для использования в конвеерном моделировании (sklearn Pipeline)
descriptors_transformer = Pipeline(steps=[('get_descriptors', 
                                           FunctionTransformer(process_molecules, validate=False))])

def get_descriptors(mols):
    return descriptors_transformer.transform(mols), descriptors_transformer


def final_data(molecules, features, target, transformer=None):

    try: 
        logger.info(f"{features}")
        logger.info(f"Количество объектов до обработки: {len(features)}")

        # дубликаты
        list_duplicates = list(features[features.duplicated()].index)
        logger.info(f"Индексы дубликатов: {list_duplicates}")
        features = features.drop(index=list_duplicates).reset_index(drop=True)
        target = target.drop(index=list_duplicates).reset_index(drop=True)


        logger.info(f"Количество объектов после удаления дубликатов: {len(features)}")

        # безопасный способ удаления элементов из списка
        # с конца удаляем элементы, с конца индексы не сдвинутся
        for index in reversed(list_duplicates):
            del molecules[index]

        # nan
        list_nan_indices = list(features[features.isnull().any(axis=1)].index)
        logger.info(f"Индексы строк с NaN: {list_nan_indices}")
        features = features.drop(index=list_nan_indices).reset_index(drop=True)
        target = target.drop(index=list_nan_indices).reset_index(drop=True)

        # Удаление NaN из molecules
        for index in reversed(list_nan_indices):
            del molecules[index]

        logger.info(f"Количество объектов после удаления NaN: {len(features)}")

    except Exception as e:
        logger.error(f"{e}")

    return molecules, features, target, transformer


def filter_target_features(features, target):
        """
        Функция для удаления редких объектов/классов из таргета и соответствующих объектов из features.
        Удаляет редкие классы для дискретных данных (<10%) и редкие значения для непрерывных данных (<5%).
        Данную функцию не нужно помещать в трансформер. Так как у данных, которые нужно будет предсказывать, не будет целевых значений
        """
        logger.info("------------------------ Запущена функция по удалению редких объектов/классов --------------------------")

        # являются ли значения дискретными числами
        if pd.api.types.is_integer_dtype(target):
            # так как датасет небольшой определим порог в 10%
            threshold_class = 0.1
            logger.info(f"Целевые значения являются дискретными числами. Уникальные значения: {set(target)}")
            value_counts = target.value_counts(normalize=True)
            rare_classes = value_counts[value_counts < threshold_class].index
            rare_indices = target[target.isin(rare_classes)].index.tolist()

            for cls, pct in value_counts.items():
                logger.info(f"Класс {cls}: {pct * 100:.2f}%")

        else: 
            # для удаления непрерывных значений используем межквартильный размах
            Q1 = np.percentile(target, 25)
            Q3 = np.percentile(target, 75)
            IQR = Q3 - Q1

            lower_threshold = Q1 - 1.5 * IQR
            upper_threshold = Q3 + 1.5 * IQR

            rare_indices = target[(target <= lower_threshold) | (target >= upper_threshold)].index.tolist()

            logger.info(f"Среднее значение в целевых переменных: {target.mean():.2f}")
            logger.info(f"Стандартное отклонение по целевым переменным: {target.std():.2f}")

        filtered_features = features.drop(index=rare_indices).reset_index(drop=True)
        filtered_target = target.drop(index=rare_indices).reset_index(drop=True)

        return filtered_features, filtered_target


# ===============================================  ПОДГОТОВКА ВСЕХ ДЕСКРИПТОРОВ =====================================================

# все дескрипторы для lasso и pca
def calc_all_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

def get_all_descriptors(mols):
    standardized_mols = standardize_molecules(mols)
    descriptors_list = [calc_all_descriptors(mol) for mol in standardized_mols]
    return pd.DataFrame(descriptors_list)


# ============================================ БАЗОВЫЕ ДЕСКРИПТОРЫ ==============================================



# ------------------------------------------------------

def final_data(features=None, target=None):

    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
    # объекты пандас на существование проверяются следующим способом: is not None
    if not isinstance(target, pd.Series):
            target = pd.Series(target)

    logger.info(f"{features}")
    logger.info(f"Количество объектов до обработки: {len(features)}")

    # удаление дубликатов
    list_duplicates = features[features.duplicated()].index.tolist()
    logger.info(f"Найдено дубликатов: {len(list_duplicates)}")
    if list_duplicates:
        features = features.drop(index=list_duplicates).reset_index(drop=True)
        target = target.drop(index=list_duplicates).reset_index(drop=True)


    logger.info(f"Количество объектов после удаления дубликатов: {len(features)}")

    # Удаление NaN
    list_nan_indices = features[features.isnull().any(axis=1)].index.tolist()
    logger.info(f"Найдено строк с NaN: {len(list_nan_indices)}")
    if list_nan_indices:
        features = features.drop(index=list_nan_indices).reset_index(drop=True)
        target = target.drop(index=list_nan_indices).reset_index(drop=True)

    logger.info(f"Количество объектов после удаления NaN: {len(features)}")

    return features, target

# функция для генерации дескрипторов из молекул
def mol_dsc_calc(mols): 
    df = DataFrame({k: f(m) for k, f in descriptors.items()} 
                     for m in mols)
    return df

descriptors_names = descriptors.keys()

# Функция-обертка для стандартизации и генерации дескрипторов
def process_base_molecules(mols, target):
    standardized_mols = standardize_molecules(mols)  # Стандартизация молекул
    features = mol_dsc_calc(standardized_mols)
    features, updated_target = final_data(features, target=target)
    return features, updated_target

def scale_data(x, title_scaler="StandardScaler"):

    if title_scaler == "StandardScaler":
        scaler = StandardScaler()
    elif title_scaler == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler choice. Use 'StandardScaler' or 'MinMaxScaler'.")
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    return x_scaled

def get_base_features(mols, target=None, scaler_type="StandardScaler"):
    def process_and_scale(mols, target):
        features, update_target = process_base_molecules(mols, target)
        scaled_features = scale_data(features, title_scaler=scaler_type)
        return scaled_features, update_target

    descriptors_transformer = Pipeline(steps=[
        ('get_scaled_descriptors', FunctionTransformer(lambda x: process_and_scale(x)[0], validate=False))
    ])

    features, updated_target = process_and_scale(mols, target)
    filtered_features, filtered_target = filter_target_features(features, updated_target)

    filtered_target = filtered_target.squeeze()

    return filtered_target, filtered_features, descriptors_transformer


#======================================= LASSO ===========================================

def process_all_features(mols, target):
    features = get_all_descriptors(mols)
    features, update_target = final_data(features=features, target=target)
    return features, update_target


def feature_selector(x, selected_indices):
    return pd.DataFrame(x).iloc[:, selected_indices]

def grid_cv_lasso(x_scaled, y, alpha_range=None, cv=5, scaler_type=None, save_dir="../lasso"):


    logger.info(f"========================== LASSO {scaler_type} для отбора признаков =========================")
    
    if alpha_range is None:
        alpha_range = np.arange(0.001, 3, 0.01)

    lasso = Lasso(max_iter=10000)
    param_grid = {'alpha': alpha_range}
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(x_scaled, y)

    optimal_alpha = grid_search.best_params_['alpha']
    logger.info(f"Оптимальный alpha: {optimal_alpha}")

    best_lasso = grid_search.best_estimator_
    coefficients = best_lasso.coef_

    selected_indices = [i for i, coef in enumerate(coefficients) if coef != 0]
    selected_features = x_scaled.iloc[:, selected_indices]
    logger.info(f"Отобранные признаки: {selected_features.columns.tolist()}")

    plt.figure(figsize=(8, 6))
    plt.plot(grid_search.cv_results_['param_alpha'], np.sqrt(-grid_search.cv_results_['mean_test_score']), label='Root Mean Test MSE', linewidth=2)
    plt.axvline(optimal_alpha, linestyle='--', color='k', label='Optimal alpha')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Root Mean Squared Error')
    plt.title(f'GridSearchCV: Выбор гиперпараметра alpha для Lasso {scaler_type}')
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Lasso_{scaler_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return optimal_alpha, selected_features

def get_lasso_features(mols, target=None, scaler_type="StandardScaler", save_dir="../lasso"):
    def process_and_scale(mols, target):
        features, updated_target = process_all_features(mols,  target)
        scaled_features = scale_data(features, title_scaler=scaler_type)
        return scaled_features,  updated_target


    scaled_features, updated_target = process_and_scale(mols, target)
    filtered_features, filtered_target = filter_target_features(scaled_features, updated_target)

    alpha, selected_features_lasso = grid_cv_lasso(
        filtered_features, filtered_target, scaler_type=scaler_type, save_dir=save_dir
    )

    selected_columns = selected_features_lasso.columns

    transformer = Pipeline(steps=[
        ('get_scaled_descriptors', FunctionTransformer(lambda x: process_and_scale(x, target)[0], validate=False)),
        ('feature_selector', FunctionTransformer(lambda x: feature_selector(x, selected_columns), validate=False)),
    ])
    
    filtered_target = filtered_target.squeeze()

    return alpha, filtered_target, selected_features_lasso, transformer


# ============================================== ОТПЕЧАТКИ МОРГАНА ======================================================

def get_morgan_features(mols, target=None, radius=3, fpSize=256):
    def calc_morgan(mols, radius=radius, fpSize=fpSize):
        # стандартизируем
        standardized_mols = standardize_molecules(mols)
        morgan_fpgenerator = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
        fingerprints = [morgan_fpgenerator.GetFingerprintAsNumPy(m) for m in standardized_mols]
        return pd.DataFrame(fingerprints)

    features = calc_morgan(mols, fpSize)
    features, updated_target = final_data(features, target=target)
    transformer = Pipeline(steps=[
        ('calc_morgan', FunctionTransformer(lambda x: calc_morgan(x, radius, fpSize), validate=False)),
        ('updated_features', FunctionTransformer(lambda x: final_data(x, target)[0], validate=False))
    ])

    filtered_features, filtered_target = filter_target_features(features, updated_target)

    filtered_target = filtered_target.squeeze()

    return filtered_features, filtered_target, transformer

# =================================================== ОТБОР ДЕСКРИПТОРОВ С ПОМОЩЬЮ PCA =========================================

def process_all_features_pca(mols, target=None):
    features = get_all_descriptors(mols)
    features, updated_target = final_data(features=features, target=target)
    return features, updated_target


def loadings_plot(coeff, feature_names, scale=1.1):

    plt.figure(figsize=(8, 8))
    
    for i in range(coeff.shape[0]):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.7, head_width=0.02)
        plt.text(coeff[i, 0] * scale, coeff[i, 1] * scale, 
                 feature_names[i], color='g', ha='center', va='center', fontsize=12)
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel("Главная компонента 1 (PC1)", fontweight='bold', fontsize=14)
    plt.ylabel("Главная компонента 2 (PC2)", fontweight='bold', fontsize=14)
    plt.title("График нагрузок для (Loadings Plot)", fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.show()


def get_pca_features(mols, target=None, n_components=0.95, scaler_type="MinMaxScaler", threshold=0.4, save_dir="../pca"):

    def process_and_scale(mols, target):
        features, updated_target = process_all_features_pca(mols, target=target)
        scaled_features = scale_data(features, title_scaler=scaler_type)
        return scaled_features, updated_target

    scaled_features, updated_target = process_and_scale(mols, target)

    # удаляем редкие данные перед pca, чтобы редкие объекты не попали в метод PCA
    filtered_features, filtered_target = filter_target_features(scaled_features, updated_target)


     # являются ли значения дискретными числами
    if pd.api.types.is_integer_dtype(target):
        type_target = "INT"
    else:
        type_target = "DOUBLE"

    logger.info("================================== PCA ======================================")
    pca = PCA(n_components=n_components)
    pca.fit(filtered_features) # сначала обучаем, чтобы посмотрет ьна колчисевто компонент

    # признаки для всех-всех фичей
    pca_components = pca.components_
    
    # маска для отбора признаков с вкладом более threshold ПО МОДУЛЮ
    significant_mask = np.abs(pca_components) > threshold
    significant_features = np.any(significant_mask, axis=0)  # Отбираем признаки, которые участвуют хотя бы в одной компоненте
    
    selected_features = filtered_features.columns[significant_features]
    x_filtered = pd.DataFrame(filtered_features, columns=filtered_features.columns).loc[:, selected_features]

    pca_result = pca.fit_transform(x_filtered) # затем уже обучаем на отфильтрованных значимых фичах
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    x_pca = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(pca_result.shape[1])])

    logger.info(f"Доля объясненной дисперсии: {explained_variance}")
    logger.info(f"Количество компонент: {pca.n_components_}")

    # filter_target_features не помещаем в трансформер
    # так как для предсказания не нужно удалять редкие таргеты (их и не будет)
    transformer = Pipeline(steps=[
        ("get_scaled_features", FunctionTransformer(lambda x: process_and_scale(x, title_scaler=scaler_type), validate=False)),
        ("pca", PCA(n_components=n_components))
    ])

    plt.rcParams['axes.linewidth'] = 1.5
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    var = np.cumsum(pca.explained_variance_ratio_) * 100  # cumulative explained variance в процентах

    plt.figure(figsize=(12, 8)) 
    plt.plot(range(1, len(var) + 1), var, 'o-', linewidth=2, markersize=6, color='b')

    # Вспомогательные линии
    plt.axhline(y=95, color='r', linestyle='--', linewidth=1, label='95% объяснения данных')
    plt.axhline(y=90, color='g', linestyle='--', linewidth=1, label='90% объяснения данных')

    # Настройка осей и меток
    plt.xticks(
        ticks=range(1, len(var) + 1, max(1, len(var) // 20)),
        labels=range(1, len(var) + 1, max(1, len(var) // 20)),
        rotation=45
    )
    plt.yticks(np.arange(0, 110, 10))
    plt.xlim(1, len(var)) 
    plt.ylim(0, 100)

    plt.xlabel('Количество главных компонент (PC)', fontsize=14, fontweight='bold')
    plt.ylabel('% объяснённой дисперсии', fontsize=14, fontweight='bold')
    plt.title(f'PCA (target={type_target}): График зависимости процентной доли информации от количества компонент: {scaler_type}', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.5)
    
    # Сохранение графика
    os.makedirs(f"{save_dir}", exist_ok=True)
    plt.savefig(f"{save_dir}/PCA_{type_target}_dispersion{scaler_type}.png", dpi=300, bbox_inches='tight')

    # x_pca - это данные после применения PCA (x_pca - это массив или DataFrame с результатами PCA)
    # pca.components_ - массив с компонентами PCA
    n_components = pca.n_components_
    n_features = pca.n_features_in_

    # выбираем нужные данные из pca.components_ (5 компонент и 7 признаков)
    data_for_heatmap = pca.components_[:, :]

    # Тепловая карта
    # Тепловая карта строится по упрощенной визуализации! Так как компонентов намного больше 
    plt.matshow(data_for_heatmap, cmap='RdBu')
    # print(pca.components_)
    # print(list(x_pca.columns))
    plt.yticks(np.arange(n_components), [f'PC{i+1}' for i in range(n_components)])
    plt.xticks(np.arange(n_features), filtered_features.columns[:n_features], rotation=60, ha='left')
    plt.colorbar()
    plt.xlabel("Характеристика")
    plt.ylabel("Главные компоненты")
    plt.title(f'PCA (target={type_target}): Heatmap - вклад некоторых признаков в некоорые компоненты: {scaler_type}')
    os.makedirs(f"{save_dir}", exist_ok=True)
    plt.savefig(f"{save_dir}/PCA_{type_target}_heatmap{scaler_type}.png", dpi=300, bbox_inches='tight')

    # График нагрузок ТОЛЬКО ДЛЯ ПЕРВЫХ ДВУХ компонентов
    # и для некоторых фичей
    # больше - график будет труднее объясняться
    feature_names = list(filtered_features.columns[:])
    loadings = np.transpose(pca.components_[:2, :])
    loadings_plot(loadings, feature_names)
    os.makedirs(f"{save_dir}", exist_ok=True)
    plt.savefig(f"{save_dir}/PCA_{type_target}_LoadingsPlot{scaler_type}.png", dpi=300, bbox_inches='tight')

    filtered_target = filtered_target.squeeze()

    return filtered_target, x_pca, transformer



def get_all_features(mols, target=None, scaler_type="StandardScaler"):
    target = target
    def process_and_scale(mols, target):
        features = get_all_descriptors(mols)
        features, updated_target= final_data(features=features, target=target)
        scaled_features = scale_data(features, title_scaler=scaler_type)
        return scaled_features, updated_target
    
    descriptors_transformer = Pipeline(steps=[
        ('get_scaled_descriptors', FunctionTransformer(lambda x: process_and_scale(x)[0], validate=False))
    ])

    features, updated_target = process_and_scale(mols, target)
    filtered_features, filtered_target = filter_target_features(features, updated_target)
    

    filtered_target = filtered_target.squeeze()

    return filtered_target, filtered_features, descriptors_transformer

