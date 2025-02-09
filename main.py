import argparse
import os
import pickle
import dill
from rdkit import Chem
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from utils.descriptors import *
parser = argparse.ArgumentParser(description='Выбор модели и обработка данных для предсказаний.')
parser.add_argument('--model', '-m', type=str, required=True, 
                    help='Выбор модели: ridge, logistic, tree')
parser.add_argument('--input', '-i', type=str, required=True, 
                    help='Входной sdf-файл с молекулой для предсказания logBCF')

parser.add_argument('--output', '-o', type=str, 
                    help='Файл .csv с предсказаниями')
args = parser.parse_args()


if os.path.isfile(args.input):
    print("file found")
else:
    print(f"Error: file {args.input} not found")
    exit(1) # завершаем программу


# Пример
with open("./linear_models/output_data/models/LogRMinMaxScNonSampl_20%.pkl", 'rb') as model_file:
    best_model = pickle.load(model_file)

# считываем sdf и превращаем в объект rdkit
molecula = [mol for mol in SDMolSupplier(args.input) if mol is not None]
output_folder = "predictions_models"
os.makedirs(output_folder, exist_ok=True)
predictions = best_model.predict(molecula)
output_df = pd.DataFrame(predictions, columns=['Predicted logBCF'])
output_df.to_csv(args.output, index=False)
print(f"Результаты предсказания сохранены в {args.output}")



