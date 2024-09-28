import requests
from tqdm import tqdm
import json
import zipfile
import os
from io import StringIO
from test_input import test_output, test_text
from collections import defaultdict

def call_api(query: str="TTE again revealed a slight decrease in LVEF", 
             api_path: str="process_text",
             api_varname: str="content",
             port: int=8000):
    url = f"http://localhost:{port}/{api_path}"
    data = {
        api_varname: {'text': query}
    }
    response = requests.post(url, json=data)
    result = response.json()
    return result

if __name__ == "__main__":
    print("Testing the api")
    anns = (call_api(query=test_text))['nlp_output']['annotations']
    print(f"Found {len(anns)} annotations in the test text")
    correct_both = 0
    for k, ann in enumerate(anns):
        concept = ann['concept_mention_string']
        start_offset = ann['start_offset']
        for ref_ann in test_output['nlp_output']['annotations']:
            c_check = ref_ann['concept_mention_string'].lower().strip() == concept.lower().strip()
            o_check = int(ref_ann['start_offset']) == int(start_offset)
            
            if c_check & o_check:
                correct_both = correct_both + 1
                continue
            
    print(f"The API has {correct_both} matches out of {len(anns)}")
     

    # check if the assets/tmp does not exist yet, otherwise continue
    if os.path.exists(os.path.join(os.getcwd(), 'assets/tmp')) == False:
        print("Starting the HFCCR_v2")   
        # Specify the path to the zip file
        zip_file_path = 'assets/HFCCR_v2.zip'

        # Extract the zip file
        print("..decompressing HFCCR_v2")  
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('assets/tmp')

    # Gather the extracted files
    print("..gathering the metadata") 
    MetaLists = {}
    for meta_file_path in tqdm(os.listdir('assets/tmp/metadata')):
        file_id = meta_file_path.split(".")[0].strip()
        with open(os.path.join('assets/tmp/metadata', meta_file_path), 'r') as file:
            lines = file.read().splitlines()
            TermList = [_l.split("-")[1].lower().strip() for _l in lines]
            MetaLists[file_id]= TermList
            
    print("..gathering the texts") 
    TextLists = {}
    for text_file_path in tqdm(os.listdir('assets/tmp/txt')):
        file_id = text_file_path.split(".")[0].strip()
        with open(os.path.join('assets/tmp/txt', text_file_path), 'r', encoding='latin1') as file:
            TextLists[file_id] = file.read()
            
    # Process the extracted files
    print("..processing the texts with the API") 
    ExtractedTerms = {}
    OutputList = []
    for Id, Text in tqdm(TextLists.items()):
        try:
            anns = (call_api(query=Text))
            anns_anns = anns['nlp_output']['annotations']
            ExtractedTerms[Id] = [ann['concept_mention_string'].lower() for ann in anns_anns]

            anns['id'] = Id

            OutputList.append(anns)
        except Exception as e:
            raise ValueError(f"Oopsydaisy something went wrong with the API: {e}, {type(anns)}")

    json_filename = 'artifacts/output.json'
    with open(json_filename, 'w') as json_file:
        json.dump(OutputList, json_file)
    # Then, zip the JSON file
    zip_filename = 'artifacts/test.json.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_filename, os.path.basename(json_filename))
    # Optionally, remove the temporary JSON file
    os.remove(json_filename)
    
    # Checking overlap
    print("..checking the overlap")
    resultDict = defaultdict(lambda:defaultdict(int))
    Averages = defaultdict(float)
    NumMissing = 0
    NumProcessed = 0
    ErrDict = defaultdict(int)
    for Id, TermList in tqdm(MetaLists.items()):
        resultDict[Id]['NumTermsInSource'] = len(TermList)
        try:        
            Extracts = ExtractedTerms[Id]
            print(set(Extracts))
            print(set(TermList))
            resultDict[Id]['NumTermsInExtraction'] = len(Extracts)
            resultDict[Id]['NumIntersection'] = len(set(TermList).intersection(set(Extracts)))
            resultDict[Id]['NumUnion'] = len(set(TermList).union(set(Extracts)))
            resultDict[Id]['IoU'] = resultDict[Id]['NumIntersection'] / resultDict[Id]['NumUnion']
            Averages['NumTermsInExtraction'] += resultDict[Id]['NumTermsInExtraction']
            Averages['NumTermsInSource'] += resultDict[Id]['NumTermsInSource']
            Averages['NumIntersection'] += resultDict[Id]['NumIntersection']
            Averages['NumUnion'] += resultDict[Id]['NumUnion']
            Averages['IoU'] += resultDict[Id]['IoU']
            NumProcessed += 1
        except Exception as e:
            ErrDict[e] += 1            
            resultDict[Id]['NumTermsInExtraction'] = 0
            resultDict[Id]['NumIntersection'] = None
            resultDict[Id]['NumUnion'] = None
            resultDict[Id]['IoU'] = None
            Averages['NumTermsInSource'] += resultDict[Id]['NumTermsInSource']
            NumMissing += 1            
            
print(f"Errors in processing: {ErrDict}")

print(f"Relative number of hits: {round(NumProcessed/(NumProcessed+NumMissing), 3)}")
print(f"Average number of extracted terms: {round(Averages['NumTermsInExtraction']/NumProcessed, 3)}")
print(f"Average number of terms in references: {round(Averages['NumTermsInSource']/(NumProcessed+NumMissing), 3)}")
print(f"Average intersection count: {round(Averages['NumIntersection']/NumProcessed, 3)}")
print(f"Average union count: {round(Averages['NumUnion']/NumProcessed, 3)}")
print(f"Average IoU: {round(Averages['IoU']/NumProcessed, 3)}")

    