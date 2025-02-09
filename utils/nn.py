import torch
from torch import Tensor, cat, mean
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from my_logger import logger
from tqdm import trange
import numpy as np
from numpy import arange
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 31012025
output_folder = "../5-nn"
os.makedirs(output_folder, exist_ok=True)


logger.info(f"================================ NN ==========================================")


# ============================================= ЗАГРУЗКА ДАННЫХ ==============================================
base_double_features_min_max_scaler = pd.read_csv("../data/learning/base_double_features_min_max_scaler.csv")
base_double_features_standard_scaler = pd.read_csv("../data/learning/base_double_features_standard_scaler.csv")
base_double_target = pd.read_csv("../data/learning/base_double_target.csv")




# ========================================== МОДЕЛИРОВАНИЕ ======================================================
class NetBCF(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return x


list_descriptors = {"MinMaxSc": base_double_features_min_max_scaler, "StdSc": base_double_features_standard_scaler}
results = []

for title, descriptors in list_descriptors.items():

    x_train, x_temp, y_train, y_temp = train_test_split(descriptors, base_double_target, test_size=0.3, random_state=RANDOM_STATE)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)
    
    # Преобразование в тензоры
    x_train, x_val, x_test = map(lambda x: torch.tensor(x.values, dtype=torch.float32), [x_train, x_val, x_test])
    y_train, y_val, y_test = map(lambda y: torch.tensor(y.values, dtype=torch.float32), [y_train, y_val, y_test])


    # загрузчик будет перемешивать данные при использовании эпохи при shuffle=True
    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                                            batch_size=10,
                                            shuffle=True)
    validation_loader = DataLoader(dataset=TensorDataset(x_val, y_val),
                                            batch_size=10,
                                            shuffle=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                                            batch_size=10,
                                            shuffle=True)

    
    N_EPOCH = 250
    IN_SIZE = descriptors.shape[1] # кол-во нейронов во входном слое
    HIDDEN_SIZE = 256 # кол-во нейронов в скрытом слое
    OUTPUT_SIZE = 1 # кол-во нейронов в выходном слое
    LEARNING_RATE = .0001 # скорость обучения

    # Создание модели
    model = NetBCF(input_dim=IN_SIZE, num_hidden=HIDDEN_SIZE, output_dim=OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # оптимизатор Adam на основе стохастического градиентного спуска
    criterion = nn.MSELoss()
    


    best_r2_val = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    r2, rmse = [], []
    
    with trange(N_EPOCH, position=0) as t:
        for epoch in t:
            t.set_description(f'EPOCH {epoch + 1}')

            # для каждой эпохи будем сохранять предсказанные и истинные значения, чтобы рассчитать 
            # коэффициент детерминации и среднеквадратичную ошибку предсказания на обучающей выборке
            pred_train = []
            true_train = []

            # в данном цикле вся обучающая выборка проходит через нейронную сеть батчами
            for x, y in train_loader:

                model.train()
                optimizer.zero_grad() # обнуление градиентов для обратного распространения ошибки
                y_pred = model(x)

                # преобразуем тензор в одномерный и добавляем его в массив
                pred_train.append(y_pred.squeeze(-1))
                true_train.append(y.squeeze(-1))

                # расчет среднеквадратичной ошибки между предсказанным и истинным значением
                loss = criterion(y_pred, y)
                # обратное распространение ошибки из переменной потери в обратном направлении через нейроную сеть
                loss.backward()
                # градиентный спуск по шагам на основе вычисленных во время операции .backward() градиентов
                optimizer.step()


            # для каждой эпохи будем сохранять предсказанные и истинные значения, чтобы рассчитать 
            # коэффициент детерминации и среднеквадратичную ошибку предсказания на тестовой выборке
            pred_val = []
            true_val = []

            for x, y in validation_loader:
            # отключим обновление весов нейронной сети.
            # таким образом, переведем нейросеть в режим оценки, 
            # чтобы проверить предсказания на валидационной выборке:
                model.eval()
                with torch.no_grad():
                    pred_val.append(model(x).squeeze(-1))
                true_val.append(y.squeeze(-1))
            

             # преобразуем тензоры в одномерные массивы
            pred_val = cat(pred_val).detach().numpy()
            true_val = cat(true_val).detach().numpy()
            pred_train = cat(pred_train).detach().numpy()
            true_train = cat(true_train).detach().numpy()

            r2_val = r2_score(true_val, pred_val)
            rmse_val = root_mean_squared_error(true_val, pred_val)
                
            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_epoch = epoch + 1  # сохраняем номер эпохи
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info(f'Net parameteres for best R2 {round(r2_val, 3)} were updated at EPOCH {epoch + 1}')

            r2.append(r2_val)
            rmse.append(rmse_val)
            t.set_postfix(r2_train=r2_score(true_train, pred_train), 
                        rmse_train=root_mean_squared_error(true_train, pred_train),
                        r2_val=r2_val, 
                        rmse_val=rmse_val)
    

    model.load_state_dict(best_model_wts)
    model.eval()
    os.makedirs(f"{output_folder}/model", exist_ok=True)
    torch.save(model.state_dict(), os.path.join(f"{output_folder}/model", f'nn_{title}.pth'))

    with torch.no_grad():
        y_pred_test = model(x_test).squeeze(-1).numpy()
    
    q2_test = r2_score(y_test.numpy(), y_pred_test)
    rmse_test = np.sqrt(root_mean_squared_error(y_test.numpy(), y_pred_test))
    
    
    results.append({
        "model": f"NetBCF_{title}",
        "optimal_epochs": best_epoch,
        "hidden_layers": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "r2_val": best_r2_val,
        "q2_test": q2_test,
        "rmse_test": rmse_test
    })
    
    # ======================================= ГРАФИКИ =============================================
    plt.figure(figsize=(15, 8))
    plt.plot(range(1, len(r2)+1), r2, c="b", label="R2")
    plt.plot(range(1, len(rmse) +1 ), rmse, c="c", label='RMSE')
    plt.legend()
    plt.xticks(arange(1, 250, 5))
    plt.yticks(arange(min(r2+rmse), max(r2+rmse)+0.05, 0.05))
    plt.scatter(r2.index(max(r2)) + 1, max(r2), c='r')
    plt.scatter(rmse.index(min(rmse)) + 1, min(rmse), c='orange')
    plt.plot([rmse.index(min(rmse)) + 1, r2.index(max(r2)) + 1], [min(rmse), max(r2)], '--', c='r')
    plt.xlabel('Количество эпох', fontsize=14)
    plt.ylabel('Значение метрик оценки качества модели (R2, RMSE)', fontsize=14)
    best_epoch = r2.index(best_r2_val) + 1
    os.makedirs(f"{output_folder}/plots", exist_ok=True)
    plt.savefig(os.path.join(f"{output_folder}/plots", f"loss_curve_{title}.png"), dpi=300)
    plt.close()
    

    # -------------------------
    pred_test = []
    true_test = []
    for x, y in test_loader:
        with torch.no_grad():
            pred_test.append(model(x).squeeze(-1))
        true_test.append(y.squeeze(-1))
    pred_test = torch.cat(pred_test).numpy()
    true_test = torch.cat(true_test).numpy()
    q2_test = r2_score(true_test, pred_test)
    prmse_test = root_mean_squared_error(true_test, pred_test)
    # с помощью нулевой модели тоже
    mean_y_train = mean(y_train)
    pred_test_null_model = Tensor([mean_y_train.item() for x in range(len(true_test))]).unsqueeze(-1)
    q2_test_null_model = r2_score(true_test, pred_test_null_model)
    prmse_test_null_model = root_mean_squared_error(true_test, pred_test_null_model)


    pred_train_null_model = Tensor([mean_y_train.item() for x in range(len(true_train))]).unsqueeze(-1)
    r2_train_null_model = r2_score(true_train, pred_train_null_model)
    rmse_train_null_model = root_mean_squared_error(true_train, pred_train_null_model)

    plt.figure(figsize=(10, 8))
    plt.scatter(pred_test, true_test, alpha=0.5, label="Внешняя контрольная выборка")
    plt.scatter(pred_train, true_train, alpha=0.2, label="Обучающая выборка")
    plt.scatter(pred_val, true_val, alpha=0.5, label="Валидационная контрольная выборка")
    plt.scatter(pred_train_null_model, true_train, alpha=0.1, label="Нулевая модель")

    plt.xlabel("Предсказанные значения LogBCF")
    plt.ylabel("Истинные значения LogBСF")

    plt.legend(loc="upper left")
    plt.plot([min(true_test), max(true_test)], [min(true_test), max(true_test)], c="b") # линия
    plt.show()
    os.makedirs(f"{output_folder}/plots", exist_ok=True)
    plt.savefig(os.path.join(f"{output_folder}/plots", f"scatter_plot_{title}.png"), dpi=300)
    plt.close()
    

table = pd.DataFrame(results)
table.to_csv(os.path.join(output_folder, "nn_results.csv"), index=False)

with open(f"{output_folder}/nn_results.txt", 'w', encoding='utf-8') as file:
    file.write(table.to_string(index=False))