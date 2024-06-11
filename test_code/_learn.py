import spacy
from spacy.training.example import Example
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Загрузка обучающих данных
with open('./data/train_data.json', 'r', encoding='utf-8') as f:
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
total_iterations = 20
total_samples = len(TRAIN_DATA)
batch_size = 200  # Определяет количество образцов, которые будут обрабатываться параллельно

def process_batch(batch):
    global optimizer
    global nlp
    global total_samples
    global batch_size

    losses = {}
    examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
    nlp.update(examples, sgd=optimizer, losses=losses, drop=0.5)
    total_samples -= len(batch)
    return losses.get("ner", 0)

with ThreadPoolExecutor() as executor:
    for itn in range(total_iterations):
        with tqdm(total=total_samples, desc=f"Iteration {itn+1}/{total_iterations}", unit="example") as pbar:
            futures = []
            batch = []
            for text, annotations in TRAIN_DATA:
                batch.append((text, annotations))
                if len(batch) == batch_size:
                    futures.append(executor.submit(process_batch, batch))
                    batch = []
            if batch:
                futures.append(executor.submit(process_batch, batch))

            for future in futures:
                pbar.update(batch_size)
                pbar.set_postfix(loss=future.result())
