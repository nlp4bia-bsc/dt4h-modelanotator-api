import csv
import os
from app.models.model_annotation import ModelAnnotation

class DictionaryLookupModel(ModelAnnotation):
    def __init__(self, csv_path):
        import spacy
        from spacy.tokens import Doc, Span
        from spacy.language import Language

        self.spacy_models = {
            "sv": "sv_core_news_sm",
            "en": "en_core_web_sm",
            "ro": "ro_core_news_sm",
            "nl": "nl_core_news_sm",
            "sp": "es_core_news_sm",
            "it": "it_core_news_sm",
            "cs": "xx_ent_wiki_sm"
        }

        self.nlp = \
            spacy.load(self.spacy_models[os.getenv("LANGUAGE", "EN").lower()])
        self.entities = self.load_entities_from_csv(csv_path)

        @Language.component("dictionary_entity_recognizer")
        def dictionary_entity_recognizer(doc):
            matches = []
            for token in doc:
                if token.lower_ in self.entities:
                    start = token.i
                    end = token.i + 1
                    entity_info = self.entities[token.lower_]
                    matches.append(Span(doc,
                                        start,
                                        end,
                                        label=entity_info["label"]))

            new_ents = []
            for span in matches:
                overlap = False
                for ent in doc.ents:
                    if span.start < ent.end and span.end > ent.start:
                        overlap = True
                        break
                if not overlap:
                    new_ents.append(span)

            doc.ents = list(doc.ents) + new_ents
            return doc

        self.nlp.add_pipe("dictionary_entity_recognizer", last=True)

    def load_entities_from_csv(self, file_path):
        entities = {}
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 5:  # Ensure there are enough columns
                    entities[row[0].lower()] = {
                        "label": row[1].lower(),
                        "dt4h_concept_identifier": row[2],
                        "nel_component_type": row[3],
                        "nel_component_version": row[4]
                    }
        return entities

    def predict(self, text, app, id: str = "", language: str = "en"):
        doc = self.nlp(text)
        annotations = []

        for ent in doc.ents:
            entity_text = ent.text.lower()
            if entity_text in self.entities:
                entity_info = self.entities[entity_text]
                annotation = {
                    "concept_class": ent.label_,
                    "start_offset": ent.start_char,
                    "end_offset": ent.end_char,
                    "concept_mention_string": ent.text,
                    "concept_confidence": 0.95,  # Example value
                    "ner_component_type": "dictionary lookup",
                    "ner_component_version": self.nlp.meta["version"],
                    "negation": "no",  # Simplified
                    "negation_confidence": 1.0,
                    "qualifier_negation": "",
                    "qualifier_temporal": "",
                    "dt4h_concept_identifier": entity_info["dt4h_concept_identifier"],
                    "nel_component_type": entity_info["nel_component_type"],
                    "nel_component_version": entity_info["nel_component_version"],
                    "controlled_vocabulary_namespace": "none",
                    "controlled_vocabulary_version": "",
                    "controlled_vocabulary_concept_identifier": "",
                    "controlled_vocabulary_concept_official_term": "",
                    "controlled_vocabulary_source": "original"
                }
                annotations.append(annotation)
            else:
                print(f"Entity '{entity_text}' not found in dictionary.")

        return self.serialize(text, annotations, id=id, language=language)
