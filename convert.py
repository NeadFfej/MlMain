import csv
import json

def convert_csv_to_spacy_format(csv_file, output_file):
    data = []

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text: str = row['processed_text']
            arr_text: list = text.split(" ")
            labels: dict = json.loads(row['target_labels_positions'].replace("'", '"'))
            entities = []

            # Создаем структуру для хранения позиций
            entities_dict = {"B-discount": [], "B-value": [], "I-value": []}
            
            """
            B-discount: 1 - позиция скидки;
            B-value: 2 - позиция значения скидки;
            I-value: 2 - позиция значения скидки (если значение не укладывается в
            один токен)
            """
            for key, value in labels.items():
                if key in entities_dict:
                    entities_dict[key].extend(value)

            # Конвертируем в формат SpaCy
            spacy_entities = []
            for label, positions in entities_dict.items():
                for pos in positions:
                    start = sum(len(arr_text[i]) + 1 for i in range(pos))  # Вычисление стартовой позиции
                    end = start + len(arr_text[pos])  # Вычисление конечной позиции
                    spacy_entities.append((start, end, label))

            data.append((text, {"entities": spacy_entities}))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Пример использования
convert_csv_to_spacy_format('./data/only_true.csv', './data/only_true.json')
