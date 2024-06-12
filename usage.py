import os
import json

import spacy

from assets import modelsT, models_dirs


# Конфиг для юзалки тут 
# =================================
# Три параметра ниже воссоздают имя модели по параметрам
MODEL_TYPE: modelsT = "large"
MODEL_TRAIN_DATA_FILE: str = "only_true.json"
MODEL_TOTAL_ITERATIONS: int = 10

# Файл для проверки корректности модели
MODEL_TEST_DATA_FILE: str = MODEL_TRAIN_DATA_FILE

# Принтует верные/неверные сравнения
NEED_TRUE_DATA: bool = False
NEED_FALSE_DATA: bool = True
# Выводит текст по которому было сравнение
NEED_STR_DATA: bool = True
# =================================


# Проверка существования модели
models_path = models_dirs.get(MODEL_TYPE)
model_name = MODEL_TRAIN_DATA_FILE.split(".")[0] + f"_{MODEL_TOTAL_ITERATIONS}for"
model_path = os.path.join(models_path, model_name)
if not os.path.exists(model_path):
    print(f"Не было найдено такой модели: {model_path}")
    exit(1)
    

# Загрузка обученной модели
nlp = spacy.load(model_path)

# Грузим словарь для проверки корректности модели
with open(os.path.join("data", MODEL_TEST_DATA_FILE), "r", encoding='utf-8') as file:
    TEST_DICT = json.loads(file.read())

true_ = 0
false_ = 0

for test_data in TEST_DICT:
    text = test_data[0]
    words = []
    
    for entity in test_data[1]["entities"]:
        words.append((text[entity[0]:entity[1]], entity[2]))

    doc = nlp(text)

    result = []
    for ent in doc.ents:
        result.append((ent.text, ent.label_))

    result = sorted(result, key=lambda x: x[1])
    words = sorted(words, key=lambda x: x[1])

    if result == words:
        true_ += 1
        if NEED_TRUE_DATA:
            if NEED_STR_DATA:
                print(text)
                print()

            print("Finded:", result)
            print("Test_f:", words)
            print("=" * 50)
    
    else:
        false_ += 1
        if NEED_FALSE_DATA:
            if NEED_STR_DATA:
                print(text)
                print()

            print("Finded:", result)
            print("Test_f:", words)
            print("=" * 50)
    

print()
print(f"True: {true_}; False: {false_}")
print(f"All: {true_+false_}")
