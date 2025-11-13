import os.path

from grobid_client_python.grobid_client import GrobidClient
import gzip
import json

client = GrobidClient(config_path="grobid_client_python/config.json")
'''this service takes as input a folder containing txt file/s where each row is an unstructured reference and 
returns TEI-annotated xml files 
'''
service_name = "processCitationList"
# choose the gz archive from Crossref
input_file = "0.json.gz"
max_lines = 100  # 1 MB

with gzip.open(input_file, 'rb') as file_gz:
    json_file = file_gz.read().decode('utf-8')

reference_list = []
json_content = json.loads(json_file)
items = json_content['items']
for item in items:
    if item.get('reference'):
        for reference in item['reference']:
            # take into account just unstructured references
            if reference.get("unstructured"):
                reference_list.append(reference['unstructured'])

# Function to write references to files

def write_references(reference_list, max_lines):
    file_index = 1
    current_line = 0
    file_path = f"unstructured/unstructured_{file_index}.txt"
    file = open(file_path, 'w')
    for reference in reference_list:
        cleaned_citation = reference.replace("\n", " ").replace("\r", " ")
        file.write(cleaned_citation + '\n')
        current_line += 1
        if current_line >= max_lines:
            file.close()
            file_index += 1
            file_path = f"unstructured/unstructured_{file_index}.txt"
            file = open(file_path, 'w')
            current_line = 0

    file.close()


# Write the references to files
if not os.path.exists("unstructured"):
    os.makedirs("unstructured")

write_references(reference_list, 100)

if not os.path.exists("structured"):
    os.makedirs("structured")


# https://grobid.readthedocs.io/en/latest/Consolidation/
client.process(service_name, input_path="unstructured",
               output="structured", n=10, verbose=True, force=True)



