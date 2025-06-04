import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
gc.enable()

# Some standard settings
plt.rcParams['figure.figsize'] = (13, 9) #(16.0, 12.0)
plt.style.use('ggplot')

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 2)

os.chdir('C://Shiv//trainings//genai//genai')
exec(open(os.path.abspath('genai_config.py')).read())

#%% GenAI: Basic call to LLM. Also known as Zero shot call to LLM

#Libraries
# pip install google-genai

# import base64
import os
from google import genai
from google.genai import types

#Few parameters
model_name = "gemini-2.0-flash" # "gemini-2.5-pro-preview-05-06"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),)
prompt_system = """Act like demographers and population scientists."""
prompt_user = """What is World population"""

#These are required for calling LLM
contents_user = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_user),],),]

generate_content_config = types.GenerateContentConfig(
    temperature=1,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    response_mime_type="text/plain",
    system_instruction=[types.Part.from_text(text=prompt_system),],)

chunk_outputs = client.models.generate_content_stream(model=model_name, contents=contents_user, config=generate_content_config,)
for chunk in chunk_outputs:
    print(chunk.text, end="")

#%% GenAI: Above in function call

def call_llm_and_get_response(prompt_system, prompt_user):
    #Few parameters
    model_name = "gemini-2.0-flash"
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),)
    
    #These are required for calling LLM
    contents_user = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_user),],),]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=prompt_system),],)
    
    chunk_outputs = client.models.generate_content_stream(model=model_name, contents=contents_user, config=generate_content_config,)

    return chunk_outputs

prompt_system = """Act like demographers and population scientists."""
prompt_user = """What is World population"""

chunk_outputs = call_llm_and_get_response(prompt_system, prompt_user)

for chunk in chunk_outputs:
    print(chunk.text, end="")

#%% GenAI: Basic call to LLM using few shots

#Libraries
# pip install google-genai


# import base64
import os
from google import genai
from google.genai import types


prompt_system = """Act like demographers and population scientists.

Here are few examples.

Question: What is polulation of World.
Answer: 8 Billion

Question: What is polulation of USA.
Answer: 0.4 Billion

Question: What is polulation of India.
Answer: 1.4 Billion

Question: What is polulation of Britain.
Answer: 0.07 Billion

"""

prompt_user = """What is New York, USA population in million unit instead of billion unit"""

chunk_outputs = call_llm_and_get_response(prompt_system, prompt_user)

for chunk in chunk_outputs:
    print(chunk.text, end="")

#%% RAG: Retrieval  Augmented generation

#We'll LangChain now onwards. It is well knwon Framework for GenAI development.

#Let us create data pdf from link # https://en.wikipedia.org/wiki/World_population

file_pdf_world_population = './data/World_population.pdf'

# http://python.langchain.com/docs/integrations/document_loaders/
# https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/

#Please install required library
# pip install -qU langchain_community pypdf

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
    
loader_pdf = PyPDFLoader(file_path = file_pdf_world_population, mode = "single") # "single" for the entire document

pdf_all_pages = loader_pdf.load()

#See top and bottom few letters
print(pdf_all_pages[0].page_content[: 500])
print(pdf_all_pages[0].page_content[-500:])

#Create Splitter object
text_char_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], #The character that should be used to split
                                    chunk_size=500, #Number of characters in each chunk
                                    chunk_overlap=50, #Number of overlapping characters between chunks
                                    )

#get chunks
chunks = text_char_splitter.split_text(pdf_all_pages[0].page_content)

#Show the number of chunks created
print(f"The number of chunks created : {len(chunks)}") #247

#See first few chunks
chunks[:3]

#### Embeddings
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# pip install -U sentence-transformers

from langchain_community.embeddings import SentenceTransformerEmbeddings

#For the first time, model will download and secondtime, it will use as is
model_embeddings = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"trust_remote_code":True}) 

embeddings = model_embeddings.embed_documents(chunks)

print(len(embeddings[0])) # 384
print(embeddings[0]) # 384

### Vector Databases:  built to handle high dimensional vectors. These databases
# specialize in indexing and storing vector embeddings for fast semantic search and retrieval.

# We will use -> Facebook AI Similarity Search (FAISS) 

#pip install faiss-cpu


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# First create index: To make searching efficient. 
# There are many types of indexes, we are going to use L2 distance search on them: IndexFlatL2.
index = faiss.IndexFlatIP(len(embeddings[0]))

#VDB object
vector_store = FAISS(
embedding_function=model_embeddings,
index=index,
docstore=InMemoryDocstore(),
index_to_docstore_id={},
)

#The 'vector_store' need follling document obkect to b einserted in VDB.
from langchain_core.documents import Document

#Transformation into Document object type
list_document_chunks = [Document(page_content=chunk, metadata={"source_file": file_pdf_world_population},) for chunk in chunks]
    
#See few
list_document_chunks[0]

#Create VDB
vector_store = FAISS.from_documents(list_document_chunks, model_embeddings)

#Save
vector_store.save_local(folder_path="./model/",index_name="vdb_World_population")

#Delete just to make sure that load VDB is working
del(vector_store)
gc.collect()

###Retrieval

# Load the VDB
vector_store = FAISS.load_local(f"./model/",index_name="vdb_World_population", embeddings = model_embeddings, allow_dangerous_deserialization = True)

#Any query
query = "When world population reached to 1 billion?"

# Perform similarity search
retrieved_documents = vector_store.similarity_search(query, k=3)

# Display results
for i, doc in enumerate(retrieved_documents):
    print(f"\nChunk {i+1}:\n{doc.page_content}")
    print("\n")

###Augmentation
retrieved_context = ''
for i, doc in enumerate(retrieved_documents):
    retrieved_context = retrieved_context + doc.page_content + "\n\n"

print(retrieved_context)

# Creating the prompt
prompt_user = f"""
Please answer the question. Please answer based on the context provided only and not from any other sources.
If the question cannot be answered based on the provided context, say 'Unable to answer'

Question: {query}

Context : {retrieved_context}
"""

###Generation

# import base64
import os
from google import genai
from google.genai import types

prompt_system = """Act like demographers and population scientists."""

chunk_outputs = call_llm_and_get_response(prompt_system, prompt_user)

for chunk in chunk_outputs:
    print(chunk.text, end="")



#https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Most%20used