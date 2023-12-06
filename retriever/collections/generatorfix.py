import ir_datasets
import os

# Load the dataset
dataset = ir_datasets.load("beir/arguana")

# Initialize dictionaries for doc_id and query_id mappings
doc_id_to_int = {}
query_id_to_int = {}

# Create 10 folders for storing documents
num_folders = 10
base_directory = os.path.join(os.getcwd(), "retriever", "collections")
for i in range(num_folders):
    folder_name = os.path.join(base_directory, str(i))
    os.makedirs(folder_name, exist_ok=True)

# Iterate over documents and write them to files in the 10 folders
for idx, doc in enumerate(dataset.docs_iter()):
    # Map each doc_id to a unique integer
    if doc.doc_id not in doc_id_to_int:
        doc_id_to_int[doc.doc_id] = len(doc_id_to_int)

    # Determine the folder number using modulo
    folder_number = idx % num_folders
    folder_path = os.path.join(base_directory, str(folder_number))
    filename = f"{doc_id_to_int[doc.doc_id]}.txt"
    file_path = os.path.join(folder_path, filename)
    # Write the document's text to a file
    # with open(file_path, 'w+', encoding='utf-8') as f:
    #     f.write(doc.text + "\n")
    
    with open(os.path.join(os.getcwd(), "retriever", "qrels-folder", "train_docs.txt"), 'a', encoding='utf-8') as file:
        file.write(f"{doc_id_to_int[doc.doc_id]} {doc.text}\n")

# # Directory for queries
# queries_directory = os.path.join(os.getcwd(), "retriever", "qrels-folder")
# os.makedirs(queries_directory, exist_ok=True)
# queries_output_file = os.path.join(queries_directory, 'queries_output.txt')

# # Write queries to a file
# with open(queries_output_file, 'w', encoding='utf-8') as file:
#     for query in dataset.queries_iter():
#         # Map each query_id to a unique integer
#         if query.query_id not in query_id_to_int:
#             query_id_to_int[query.query_id] = len(query_id_to_int)
#         file.write(f"{query_id_to_int[query.query_id]} {query.text}\n")

# print(f"Queries written to {queries_output_file}")

# # Directory for qrels
# qrels_directory = os.path.join(os.getcwd(), "retriever", "qrels-folder")
# os.makedirs(qrels_directory, exist_ok=True)
# qrels_output_file = os.path.join(qrels_directory, 'qrels_output.txt')

# # Write qrels to a file
# with open(qrels_output_file, 'w', encoding='utf-8') as file:
#     for qrel in dataset.qrels_iter():
#         # Map each query_id to a unique integer
#         if qrel.query_id not in query_id_to_int:
#             query_id_to_int[qrel.query_id] = len(query_id_to_int)

#         # Map each doc_id to a unique integer
#         if qrel.doc_id not in doc_id_to_int:
#             doc_id_to_int[qrel.doc_id] = len(doc_id_to_int)

#         file.write(f"{query_id_to_int[qrel.query_id]} {doc_id_to_int[qrel.doc_id]} {qrel.relevance}\n")

# print(f"Qrels written to {qrels_output_file}")
