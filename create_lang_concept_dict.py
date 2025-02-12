import csv
import os

entities = []

language = os.getenv("DICTIONARY_FILE", "english")

language_source_files_path = "./lang/"

with open(os.path.join(language_source_files_path, f"{language}_variables.csv"),
          "r") as file:
    reader = csv.reader(file)
    for row in reader: 
        entities.append([row[1], "", row[0], "DT4H", "2024-06-01"])

with open(os.path.join(language_source_files_path, f"{language}_medication.csv"),
          "r") as file:
    reader = csv.reader(file)
    for row in reader:
        entities.append([row[1], "Medication", row[0], "DT4H", "2024-06-01"])

with open(f"./{language}_entities.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader: 
        for idx in range(len(entities)):
            if entities[idx][2] == row[2]:
                if entities[idx][1] == "":
                    entities[idx][1] = row[1]

with open(f"./{language}_entities.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(entities)
