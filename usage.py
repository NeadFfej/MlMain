import spacy
import json


# Загрузка обученной модели
nlp = spacy.load("./data/discount_model_only_true_30or_sm")

# Текст для предсказания


with open("./data/only_true.json", "r", encoding='utf-8') as file:
    all_dict = json.loads(file.read())

true_ = 0
false_ = 0

for test_data in all_dict:
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

    print(result)
    print(words)
    if result == words:
        true_ += 1
    else:
        false_ += 1
    print("="*10)


print()
print(f"True: {true_}; False: {false_}")
print(f"All: {true_+false_}")
    
