#!/bin/bash

echo "Hello! You ran my script :)"

echo "$(date): Script started" >> script.log


MODEL_PATH=$1
INPUT_FILE=$2
OUTPUT_FILE=$3

python3 main.py --model "$MODEL_PATH" --input "$INPUT_FILE" --output "$OUTPUT_FILE"

OUTPUT_DIR="predictions_models"
mkdir -p "$OUTPUT_DIR"


# на сервере не могу запускать под sudo и проверить тоже не могу
# дайте доступ к sudo ! :)
# docker run --rm \
#   -v "$(pwd)/$MODEL_PATH:/opt/model" \
#   -v "$(pwd)/$INPUT_FILE:/opt/input" \
#   -v "$(pwd)/$OUTPUT_DIR:/opt/output" \
#   my_image /opt/model/model.pkl /opt/input/input.sdf /opt/output/$OUTPUT_FILE


