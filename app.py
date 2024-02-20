import streamlit as st
import sqlite3
from hashlib import sha256
import streamlit as st
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chains.llm import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from datetime import date

# Create a SQLite database and table
conn = sqlite3.connect("user_credentials.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')
conn.commit()


if 'embeddings' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
)
def get_similar_docs(query):
    db = FAISS.load_local('faiss_index',st.session_state.embeddings)
    docs = db.similarity_search_with_score(query,100)
    return docs
def format_docs(docs):
        return " ".join(doc.page_content for doc in docs)
def get_advice_from_llm(query):
    db = FAISS.load_local(st.session_state.username,st.session_state.embeddings)  
    retriever = db.as_retriever()
    llm = LlamaCpp(model_path="./tinyllama-1.1b-chat-v1.0.Q8_0.gguf",n_ctx = 2048)
    chat_history_str = "\n".join(["<|im_start|>" + entry[0]+ entry[1] +"<|im_emd|>\n" for entry in st.session_state['chat_history']])

    template = """" 
        <|im_start|>system
        {context}""" +  chat_history_str + "<|im_end|>"\
        """
        <|im_start|>user{input}<|im_end|>

        <|im_start|>assistant
       """    
    
    prompt = PromptTemplate(input_variables=["input","context"], template=template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    rag_chain = ( {"context": retriever|format_docs, "input": RunnablePassthrough()}| llm_chain)
    answer = rag_chain.invoke(query)
    return answer

def vectordb_entry():
    loader = TextLoader(f"./{st.session_state.username}.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)
    db = FAISS.load_local(st.session_state.username,st.session_state.embeddings)
    db.add_documents(docs)
    db.save_local(st.session_state.username)

def save_into_text_file(file_path,text):
    with open(file_path, 'w') as file:
        file.write(text)
    print(f"String saved to {file_path}")

def journal():
   
    messages = st.container(height=600)
    query = st.chat_input("Need some advice?")

    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if query:
        answer = get_advice_from_llm(query)
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer['text']))
        st.session_state.input_key += 1

    if 'chat_history' in st.session_state and st.session_state.chat_history:
        for speaker, message in st.session_state.chat_history:
            if speaker == "user":
                who = "You"
            else:
                who = "JournaLLM"

            messages.chat_message(speaker).write(who + ': '+ str(message))

    if st.button('Reset Chat'):
        st.session_state.chat_history = []
        st.session_state.input_key += 1
        st.experimental_rerun()


# Function to hash passwords
def hash_password(password):
    return sha256(password.encode()).hexdigest()

# Function to check login credentials
def authenticate(username, password):
    hashed_password = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    return cursor.fetchone() is not None

# Function to add a new user to the database
def add_user(username, password):
    hashed_password = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True  # User added successfully
    except sqlite3.IntegrityError:
        return False  # Username already exists

# Streamlit Login Page
def login_page():
    st.title("Login Page")
    un = st.text_input("Username:")
    pw = st.text_input("Password:", type="password")
    if un and pw:
        st.session_state['username'] = un
        st.session_state['password'] = pw
    
    if st.button("Login"):
        if not st.session_state['username'] or not st.session_state['password']:
            st.error("Both username and password are required.")
        elif authenticate(st.session_state['username'], st.session_state['password']):
            create_table()
            st.success("Login successful!")
        else:
            st.error("Invalid credentials. Please try again.")

# Streamlit Signup Page
def signup_page():
    st.title("Signup Page")
    new_username = st.text_input("New Username:")
    new_password = st.text_input("New Password:", type="password")

    if st.button("Signup"):
        if not new_username or not new_password:
            st.error("Both username and password are required.")
        else:
            result = add_user(new_username, new_password)
            if result:
                file_path = f"{new_username}.txt"
                text = "I've started writing my journal"
                # Open the file in write mode and write the string
                with open(file_path, 'w') as file:
                    file.write(text)

                print(f"String saved to {file_path}")
                loader = TextLoader(f"./{new_username}.txt")
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.split_documents(documents)
                embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={"device": "cpu"},
                )
                db = FAISS.from_documents(docs,embeddings)
                db.save_local(new_username)
                st.success("Signup successful! You can now login.")

            else:
                st.error("Username already exists. Please choose a different username.") 


def create_table():
    conn = sqlite3.connect(f'{st.session_state.username}_entries.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            notes TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Function to insert data into the SQLite database
def insert_data(date, notes):
    conn = sqlite3.connect(f'{st.session_state.username}_entries.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO entries (date, notes)
        VALUES (?, ?)
    ''', (date, notes))

    conn.commit()
    conn.close()

# Function to retrieve data for a selected date
def retrieve_data(selected_date):
    conn = sqlite3.connect(f'{st.session_state.username}_entries.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT date, notes FROM entries WHERE date = ?
    ''', (selected_date,))

    data = cursor.fetchall()

    conn.close()
    return data


def entry():
    st.title('JournaLLM')
    st.write('Welcome to JournaLLM, \
             your personal space for mindful \
             reflection and goal tracking! This app is designed to help you \
             seamlessly capture your daily thoughts, \
             set meaningful goals, and track your progress.')
    c1,c2 = st.columns(2)
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    file_path = f"{st.session_state.username}.txt"

    c1.write("Today's Entry")
    text0 = c1.text_area("Enter text ")
  
    # template = f'''Question: What happened on {date.today().strftime("%B %d, %Y")}? 
    # How did I feel on {date.today().strftime("%B %d, %Y")}? 
    # What were the events that happened on {date.today().strftime("%B %d, %Y")}? 
    # Describe your day, {date.today().strftime("%B %d, %Y")}. \n Answer: '''
    text = f""" <|im_start|>system
         What happened on {date.today().strftime("%B %d, %Y")}? 
    How did I feel on {date.today().strftime("%B %d, %Y")}? 
    What were the events that happened on {date.today().strftime("%B %d, %Y")}? 
    Describe your day, {date.today().strftime("%B %d, %Y")}.<|im_end|>
    
        <|im_start|>user
        {text0}<|im_end|>"""


    if c1.button('Pen down') and text:
        save_into_text_file(file_path,text)
        vectordb_entry()
        c1.write('Entry saved')
        st.session_state.input_key += 1
        #display previous entries
        insert_data(date.today().strftime("%B %d, %Y"), text0)

    #displaying 
    c2.write('View previous entries')
    selected_date = c2.date_input('Select a date', date.today())
    data = retrieve_data(selected_date.strftime("%B %d, %Y"))
    if data:
        en = c2.container(height=300)
        for i in data:
            en.write(i[1])
        #[en.write(x[1]) for x in data]
    else:
        c2.info('No entries for the selected date.')



# Main Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Login", "Signup","Journal","Advice"])

    if page == "Login":
        login_page()
    elif page == "Signup":
        signup_page()
    elif page == "Journal":
        if 'username' not in st.session_state:
            st.write('Please login to continue.')
        else:
            st.write(f"Logged in as {st.session_state.username}")
            entry()
    elif page == "Advice":
        if 'username' not in st.session_state:
            st.write('Please login to continue.')
        else:
            st.write(f"Logged in as {st.session_state.username}")
            journal()

if __name__ == "__main__":
    main()

