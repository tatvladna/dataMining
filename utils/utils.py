import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import (root_mean_squared_error, r2_score, f1_score, balanced_accuracy_score,
                            roc_auc_score, roc_curve, confusion_matrix)
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import psutil
from xgboost import plot_importance
from my_logger import logger
import dill
import seaborn as sns

# ============================================  Settings ======================================================
# Настройки pandas`a
def start():
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 35,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+


# =========================================== GridSearchCV =========================================================

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss 
    return mem / (1024 ** 2)  # Перевод в МБ

def grid_cv(model, title, param_grid, descriptors, scoring, target,
            size, state, output_folder=None, model_name=None, 
            task='regression', sampling = "NonSampl", transformer=None):
    
    x_train, x_test, y_train, y_test = train_test_split(descriptors, target, test_size=size, random_state=state)

    # преобразование в одномерный массив
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


    if model_name=="Ridge":
        title = model_name + title + f"_{size:.0%}"

    elif model_name=="SVMClsf":
        
        activity_counts_train = pd.Series(y_train).value_counts().to_dict()
        activity_counts_test = pd.Series(y_test).value_counts().to_dict()
        logger.info(f"activity_counts_train: {activity_counts_train}")
        logger.info(f"activity_counts_test: {activity_counts_test}")
        if sampling == "SMOTE":
            logger.info(f"Размеры до SMOTE: {x_train.shape}, {len(y_train)}")
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            logger.info(f"Размеры после SMOTE: {x_train.shape}, {len(y_train)}")
        elif sampling == "UnderSampl":
            logger.info(f"Размеры до UnderSampling: {x_train.shape}, {len(y_train)}")
            rus = RandomUnderSampler(random_state=42)
            x_train, y_train = rus.fit_resample(x_train, y_train)
            logger.info(f"Размеры после UnderSampling: {x_train.shape}, {len(y_train)}")
        elif sampling == "SMOTETomek":
            logger.info(f"Размеры до SMOTETomek: {x_train.shape}, {len(y_train)}")
            smote_tomek = SMOTETomek(random_state=42)
            x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
            logger.info(f"Размеры после SMOTETomek: {x_train.shape}, {len(y_train)}")
            
        title = model_name + sampling + title
    
    else:
        title = model_name + title
    logger.info(f"========================== Model: {title}   ===============================")
    """
    GridSearchCV: 
        1. Возвращает метрики, графики обученных моделей
        2. Сохраняет наилучшую модель и трансформер в файлы формата .pkl.
    """


    
    # лес дольше обучаетсяс фолдами
    # RepeatedKFold, KFold, StratifiedKFold
    # rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42) # он же просто cv

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        cv=kf,
        n_jobs=4
        # return_train_score=True # возвращаем метрику на тренировочных данных
    )
    logger.info(f".........Идет обучение ............")
    start_t = time.time()
    start_memory = get_memory_usage()
    try:
        grid.fit(x_train, y_train)
    except Exception as e:
        logger.exception("Ошибка при обучении модели: %s", e)
        exit(1)
    end_t = time.time()
    end_memory = get_memory_usage()
    total_time = end_t - start_t
    minutes = int(total_time // 60)
    seconds = total_time % 60
    logger.info(f"Суммарное время обучения: {minutes} мин {seconds:.2f} сек")

    # количество моделей, обученных GridSearchCV (это количество комбинаций гиперпараметров = количеству моделей)
    num_models_trained = len(grid.cv_results_['params'])
    # суммарный объем ram делим на количество моделей = средний объем ram на обучение одной модели
    memory_used = (end_memory - start_memory) / num_models_trained
    # best_index = grid.best_index_
    # n_splits = getattr(grid.cv, 'n_splits', 1)
    # n_repeats = getattr(grid.cv, 'n_repeats', 1)  # 1, если n_repeats не используется
    
    # ======  ЛУЧШАЯ МОДЕЛЬ ========
    best_model = grid.best_estimator_
    # ===============================

    logger.info(f"..........Сохранение модели в pipeline.........")
    pipeline = Pipeline([
    ('scaler', transformer),  # Добавляем трансформер
    ('model', best_model)  # Добавляем модель
    ])



    os.makedirs(f"{output_folder}/models", exist_ok=True)
    pipeline_filename = f"{output_folder}/models/{title}.pkl"

    with open(pipeline_filename, 'wb') as model_file:
        dill.dump(pipeline, model_file)

    model_size = os.path.getsize(pipeline_filename) / (1024 ** 2)  # Размер модели в МБ

    logger.info(f"Лучшая модель сохранена в: {pipeline_filename}")
    logger.info(f'|best params|: {grid.best_params_}')
    logger.info(f'|best fit time|: {grid.refit_time_}')
    y_pred = grid.predict(x_test)
    # logger.info(f"Суммарное время предсказания c перекрестной проверкой (сек): {grid.cv_results_['mean_score_time'][grid.best_index_] * n_splits * n_repeats}")

    if task=="regression":
        rmse = root_mean_squared_error(y_test, y_pred)
        # best_index = grid.best_index_ # индекс лучшей комбинации
        y_train_pred = grid.best_estimator_.predict(x_train)
        r2 = r2_score(y_train, y_train_pred) # на трейне
        # r2 = grid.best_score_ # r2 на кросс-валидации, либо: grid.cv_results_['mean_test_score'][grid.best_index_]
        y_test_pred = grid.best_estimator_.predict(x_test)
        q2_test = r2_score(y_test, y_test_pred) # на тесте

        logger.info(f'R2 on test set: {r2}')
        logger.info(f'RMSE on test set: {rmse}')

    # -------------------- индивидуальные графики ------------------------
    if model_name=="Ridge":
        # иначе не заработает
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        # эталонная линия
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title}: True vs Predicted')
        # создаем папку, если она не существует
        os.makedirs(f'{output_folder}/plot', exist_ok=True)
        plt.savefig(f'{output_folder}/plot/{title}.png', dpi=300, bbox_inches='tight')
        plt.close()

    elif model_name == "ForestRegr":
        if not title.endswith("MorganFP256"):
            # Важность признаков
            plt.figure(figsize=(10, 6))
            feature_importances = best_model.feature_importances_
            
            # выведем только первые 10, которые вносят наибольший вклад
            # для читаемости графика
            feature_importances_df = pd.DataFrame({
                'Feature': x_train.columns,
                'Importance': feature_importances
            })
            
            top_features = feature_importances_df.sort_values(by='Importance', ascending=False).head(10)
            
            plt.barh(top_features['Feature'], top_features['Importance'])
            plt.xlabel('Важность признаков')
            plt.ylabel('Признаки')
            plt.title(f'Топ-10 важнейших признаков для модели: {title}')

            os.makedirs(f'{output_folder}/importances', exist_ok=True)
            plt.savefig(f'{output_folder}/importances/{title}', dpi=300, bbox_inches='tight')
            plt.close()
        
    elif model_name=="XGBRegr":
        if not title.endswith("MorganFP256"):
            plt.figure(figsize=(10, 6))
            plot_importance(best_model, importance_type='weight', max_num_features=10)  # max_num_features — ограничивает количество признаков
            plt.title(f'Важность признаков по версии XGBRegressor для модели: {title}')
            os.makedirs(f'{output_folder}/importances', exist_ok=True)
            plt.savefig(f'{output_folder}/importances/{title}', dpi=300, bbox_inches='tight')
            plt.close()

    elif model_name=="KNNRegr":
        n_neighbors = param_grid['n_neighbors']
        mean_scores = -grid.cv_results_['mean_test_score']

        plt.figure(figsize=(10, 5))
        plt.plot(n_neighbors, mean_scores, marker='o', linestyle='dashed', color='b')
        plt.xlabel('Число соседей')
        plt.ylabel('Среднеквадратичная ошибка')
        plt.title('Выбор оптимального числа соседей в KNN-регрессии')
        plt.grid()
        os.makedirs(f'{output_folder}/plot', exist_ok=True)
        plt.savefig(f'{output_folder}/plot/{title}.png', dpi=300, bbox_inches='tight')
        plt.close() # обязательно закрываем график


    elif model_name=="SVMClsf":
            
            # ==================  Метрики на тестовых данных  ===========================
            y_probs = grid.predict_proba(x_test)[:, 1] # вероятность для положительного класса
            f1 = f1_score(y_test, y_pred, average='weighted')
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_probs)

            logger.info(f'F1 Score: {f1}')
            logger.info(f'Balanced Accuracy: {balanced_acc}')
            logger.info(f'AUC: {auc}')

            # ==========================  Confusion Matrix  ===========================
            conf_matrix = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Class 1', 'Class 3'], yticklabels=['Class 1', 'Class 3'],
                        annot_kws={"size": 18})
            plt.xlabel('Predicted Label', fontsize=18)
            plt.ylabel('True Label', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            
            os.makedirs(f'{output_folder}/matrix', exist_ok=True)
            plt.savefig(f'{output_folder}/matrix/{title}.png', dpi=300, bbox_inches='tight')
            plt.close() # обязательно закрываем график

            # =========================  ROC Curve  ===================================
            
            # у нас были класс 1 и 3, для построение roc-кривой должны быть бинарные метри
            # указываем 1 как отрицатльнй класс, а 3 - положительный 
            fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=3)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve {title} (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='best')

            os.makedirs(f'{output_folder}/roc', exist_ok=True)
            plt.savefig(f'{output_folder}/roc/{title}.png', dpi=300, bbox_inches='tight')
            plt.close() # обязательно закрываем график

            # sensitivity = tpr # - список значений
            # specificity = 1 - fpr # cписок значений




            return round(f1, 2),\
            round(balanced_acc, 2),\
            round(auc, 2),\
            round(grid.refit_time_, 2),\
            grid.best_params_,\
            round(model_size, 2),\
            round(memory_used, 2),\
            activity_counts_train,\
            activity_counts_test

    # среднее время предсказания на всех валидационных данных (для всех фолдов в каждой итерации)
    # grid.cv_results_['mean_score_time'][grid.best_index_]
    # grid.refit_time_ - время, которое затрачивается на обучение модели после того, как все фолды прошли, и лучший набор гиперпараметров был выбран.
    return  round(r2, 2),\
            round(q2_test, 2), \
            round(rmse, 2), \
            round(grid.refit_time_, 2), \
            grid.best_params_, \
            round(memory_used, 2),\
            round(model_size, 2)

