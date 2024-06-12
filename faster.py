import spacy
from spacy.training.example import Example
import json
from tqdm import tqdm

# spacy.load('ru_core_news_sm')
spacy.load('ru_core_news_lg/ru_core_news_lg/ru_core_news_lg-3.7.0')

# Загрузка обучающих данных
with open('./data/only_true.json', 'r', encoding='utf-8') as f:
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

# Параметры обучения
total_iterations = 1000
batch_size = 100

# Обучение модели
total_samples = len(TRAIN_DATA)
for itn in range(total_iterations):
    losses = {}
    with tqdm(total=total_samples, desc=f"Iteration {itn + 1}/{total_iterations}", unit="example") as pbar:
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_data = TRAIN_DATA[batch_start:batch_end]
            batch_examples = []
            for text, annotations in batch_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                batch_examples.append(example)
            nlp.update(batch_examples, sgd=optimizer, losses=losses)
            pbar.update(len(batch_data))
        pbar.set_postfix(loss=losses.get("ner", 0))

# Сохранение модели
nlp.to_disk("./data/discount_model_all_20for_lg")