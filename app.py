import streamlit as st
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 
st.session_state.embeddings = HuggingFaceEmbeddings()
def get_similar_docs(query):
    db = FAISS.load_local('faiss_index',HuggingFaceEmbeddings())
    docs = db.similarity_search_with_score(query,100)
    return docs

def vectordb_entry():
    loader = TextLoader("./output.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = FAISS.load_local('faiss_index',st.session_state.embeddings)
    db.add_documents(docs)
    db.save_local('faiss_index')

def save_into_text_file(file_path,text):
    with open(file_path, 'w') as file:
        file.write(text)
    print(f"String saved to {file_path}")

def main():
    st.title('JournaLLM')
    file_path = "output.txt"
    text = st.text_area("Today's Entry")
    if st.button('Pen down') and text:
        save_into_text_file(file_path,text)
        vectordb_entry()
        st.write('Entry saved')
    query = st.text_input('Query')
    
    if st.button('Get similar entries'):
        docs = get_similar_docs(query)
        st.text('similar docs in db')
        #st.write(docs)
        context = []
        for i in docs:
            context.append(i[0].page_content)
        st.write(context)

if __name__=="__main__":
    main()