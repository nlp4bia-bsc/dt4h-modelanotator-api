from abc import ABC, abstractmethod
from datetime import datetime
import json

class ModelAnnotation(ABC):
    @abstractmethod
    def predict(self, text, app, id):
        pass

    def serialize(self, text, annotations, id, language="en"):
        """This function implements the Common Data Model v2"""
        output = {
            "nlp_output": {
                "record_metadata": {
                    "clinical_site_id": "example_site",
                    "patient_id": "patient_id",
                    "admission_id": "",
                    "record_id": "patient_id",
                    "record_type": "progress report",
                    "record_format": "txt",
                    "record_creation_date": datetime.now().isoformat(),
                    "record_lastupdate_date": datetime.now().isoformat(),
                    "record_character_encoding": "UTF-8",
                    "record_extraction_date": datetime.now().isoformat(),
                    "report_section": "",
                    "report_language": language,
                    "deidentified": "no",
                    "deidentification_pipeline_name": "",
                    "deidentification_pipeline_version": "",
                    "text": text,
                    "id": str(id),
                    "nlp_processing_date": datetime.now().isoformat(),
                    "nlp_processing_pipeline_name": self.__class__.__name__,
                    "nlp_processing_pipeline_version": "1.0",
                },
                "annotations": annotations
            },
            "nlp_service_info": {
                "service_app_name": "DT4H NLP Processor",
                "service_language": language,
                "service_version": "1.0",
                "service_model": self.__class__.__name__
            }
        }
        output["nlp_output"]["processing_success"] = True
        return output
