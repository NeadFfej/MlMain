import spacy


def func(text, model):
    nlp = spacy.load(model)
    mass = text.split(" ")
    res = ['O' for i in range(len(mass))]
    doc = nlp(text)
    for ent in doc.ents:
        res[ent.start] = ent.label_
    return str(res)


s = input("Введите текст\n")
model = input("Введите название модели\n")
ans = func(s, model)
print(ans)
