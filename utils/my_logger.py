import logging
import os

# относительный путь (относитльно папки /utils)
log_filename = os.path.join(os.path.dirname(__file__), 'logs', 'logger.log')
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# # для консоли
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# logger.info("Загрузка данных...")
# logger.warning("Предупреждение!")
# logger.error("Ошибка!")
