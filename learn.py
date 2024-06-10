import spacy
from spacy.training.example import Example
import json
from tqdm import tqdm


spacy.load('ru_core_news_sm')
#spacy.load('ru_core_news_lg')

# Загрузка обучающих данных
#with open('./data/train_data.json', 'r', encoding='utf-8') as f:
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

# Обучение модели
total_iterations = 10
total_samples = len(TRAIN_DATA)
for itn in range(total_iterations):
    losses = {}
    with tqdm(total=total_samples, desc=f"Iteration {itn+1}/{total_iterations}", unit="example") as pbar:
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
            pbar.update(1)
        pbar.set_postfix(loss=losses.get("ner", 0))

# Сохранение модели
nlp.to_disk("./data/discount_model_only_true")
