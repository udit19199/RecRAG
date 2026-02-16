from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("./data/pdfs").load_data()

vector_index = VectorStoreIndex.from_documents(documents)
vector_index.as_query_engine()

vector_index.storage_context.persist(persist_dir="./storage/")
