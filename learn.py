from copy import copy
import json
import os

import spacy
from spacy.training.example import Example
from tqdm import tqdm

from assets import modelsT, models_names, models_dirs


for model_size, model_name in copy(models_names).items():
    try:
        __import__(model_name)
    except ImportError:
        models_names.pop(model_size)
        print(f"У вас не установленна модель `{model_name}` из общего списка")
        print(f"Попробуйте python -m spacy download {model_name}")
        print("Или добавьте данную модель как пакет к проекту вручную")
        print("=" * 50)


# Конфиг для обучалки тут
# =================================
MODEL_TYPE: modelsT = "small"
TRAIN_DATA_FILE: str = "only_true.json"

# Количество итераций обучения
TOTAL_ITERATIONS: int = 1
# Больше 10 на only_true не имеет смысла
#  для других объёмов входных данных нужно тестировать
BATCH_SIZE: int = 10
# =================================


# Загружаем предобученную модель
if model := models_names.get(MODEL_TYPE, None):
    spacy.load(model)
else:
    print("У вас нет выбранной модели!")
    exit(1)

# Загрузка обучающих данных
with open(os.path.join("data", TRAIN_DATA_FILE), "r", encoding="utf-8") as f:
    TRAIN_DATA = json.load(f)

# Создание пустой модели
nlp = spacy.blank("ru")

# Добавление компонента NER
ner = nlp.add_pipe("ner", last=True)

# Добавление меток
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Начало обучения
optimizer = nlp.begin_training()

# Обучение модели
total_samples = len(TRAIN_DATA)
for itn in range(TOTAL_ITERATIONS):
    losses = {}
    with tqdm(
        total=total_samples,
        desc=f"Iteration {itn + 1}/{TOTAL_ITERATIONS}",
        unit="example",
    ) as pbar:
        for batch_start in range(0, total_samples, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_samples)
            batch_data = TRAIN_DATA[batch_start:batch_end]
            batch_examples = []
            for text, annotations in batch_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                batch_examples.append(example)
            
            nlp.update(batch_examples, sgd=optimizer, losses=losses)
            pbar.update(len(batch_data))
        
        pbar.set_postfix(loss=losses.get("ner", 0))


# Сохранение модели
models_path = models_dirs.get(MODEL_TYPE)
base_model_name = TRAIN_DATA_FILE.split(".")[0] + f"_{TOTAL_ITERATIONS}for"
model_name = base_model_name
version = 0

while os.path.exists(os.path.join(models_path, model_name)):
    version += 1
    model_name = f"{base_model_name} ({version})"

nlp.to_disk(os.path.join(models_path, model_name))
