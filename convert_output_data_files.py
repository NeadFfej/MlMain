import spacy
import csv


def func(file, file1, model):
    nlp = spacy.load(model)
    csvfile = open(file, "r", encoding="utf8", newline="")
    csvfile1 = open(file1, "w", encoding="utf8", newline="")
    writer = csv.DictWriter(csvfile1, fieldnames=['processed_text', 'label'])
    writer.writeheader()
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row["processed_text"]
        mass = text.split(" ")
        res = ['O' for i in range(len(mass))]
        doc = nlp(text)
        for ent in doc.ents:
            res[ent.start] = ent.label_
        writer.writerow({'processed_text': text, 'label': str(res)})


model = input("Введите название модели\n")
func("./data/gt_test.csv", "./data/ans_gt_test.csv", model)
