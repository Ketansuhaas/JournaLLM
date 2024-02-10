import streamlit as st
import sqlite3
from hashlib import sha256
import streamlit as st
from datetime import datetime
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


# Create a SQLite database named 'user_credentials.db'. If it already exists, connect to it.
# This database is used for storing user credentials.
conn = sqlite3.connect("user_credentials.db")

# Create a cursor object using the connection. 
# The cursor is used to execute SQL commands.
cursor = conn.cursor()

# Execute an SQL command using the cursor. 
# This command attempts to create a new table named 'users' if it doesn't already exist.
# The table is designed to store usernames and passwords.
# It has two columns: 'username' and 'password'.
# 'username' is of type TEXT and is set as the PRIMARY KEY, ensuring that each username is unique.
# 'password' is also of type TEXT to store the password associated with each username.
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')

# Commit the changes made by the cursor.execute command to the database.
# This ensures that the creation of the table is saved in the database.
conn.commit()


# Check if the embeddings model is not already stored in Streamlit's session state
if 'embeddings' not in st.session_state:
    # Initialize and store the embeddings model in session state for efficient reuse
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Specify the model name to use
        model_kwargs={"device": "cpu"},  # Force the model to run on CPU
    )


def get_similar_docs(query):
    # Load the FAISS index from local storage using the embeddings model stored in session state
    db = FAISS.load_local('faiss_index', st.session_state.embeddings)
    # Perform a similarity search in the FAISS database for the given query, returning top 100 similar documents with scores
    docs = db.similarity_search_with_score(query, 100)
    # Return the list of similar documents and their similarity scores
    return docs

def format_docs(docs):
    # Join the page content of each document in docs with a space, and return the concatenated string
    return " ".join(doc.page_content for doc in docs)

def get_advice_from_llm(query):
    # Load the FAISS index and initialize it with embeddings from session state for document retrieval
    db = FAISS.load_local('faiss_index', st.session_state.embeddings)
    # Convert the FAISS index into a retriever for fetching documents based on queries
    retriever = db.as_retriever()
    # Initialize the LlamaCpp model with specified model path and context size
    llm = LlamaCpp(model_path="./tinyllama-1.1b-chat-v1.0.Q8_0.gguf", n_ctx=2048)
    # Create a chat history string from session state for inclusion in the prompt template
    chat_history_str = "\n".join(["" + entry[0] + entry[1] + "\n" for entry in st.session_state['chat_history']])
    # Define the prompt template with placeholders for dynamic context and user input
    template = """" 
        system
        {context}""" + \
        chat_history_str +\
        """
        user{input}

        assistant
       """    
    # Initialize a PromptTemplate with variables for dynamic insertion into the template
    prompt = PromptTemplate(input_variables=["input", "context"], template=template)
    # Chain the LlamaCpp model with the prompt for generating responses
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define the RAG chain combining retriever and LLM chain for generating advice based on the query
    rag_chain = ({"context": retriever | format_docs, "input": RunnablePassthrough()} | llm_chain)
    # Invoke the RAG chain with the user query to get advice
    answer = rag_chain.invoke(query)
    # Return the generated advice
    return answer


def vectordb_entry():
    # Load text documents from a specified file
    loader = TextLoader("./output.txt")
    documents = loader.load()
    # Split loaded documents into smaller chunks for vectorization
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)
    # Load or create a FAISS vector database using embeddings from session state
    db = FAISS.load_local('faiss_index', st.session_state.embeddings)
    # Add the document chunks to the FAISS database
    db.add_documents(docs)
    # Save the updated database locally
    db.save_local('faiss_index')


def save_into_text_file(file_path, text):
    # Open the specified file path for writing and save the provided text string into it
    with open(file_path, 'w') as file:
        file.write(text)
    # Print a confirmation message indicating where the text was saved
    print(f"String saved to {file_path}")


def journal():
    # Create a container to display chat messages with a specified height
    messages = st.container(height=600)
    # Create a chat input box for users to enter their queries
    query = st.chat_input("Need some advice?")

    # Initialize an input_key in session state if it doesn't exist, to track input changes
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    # Initialize chat_history in session state if it doesn't exist, to store chat interactions
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # If the user has entered a query
    if query:
        # Get advice from the language model for the given query
        answer = get_advice_from_llm(query)
        # Append the user's query and the model's response to the chat history
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer['text']))
        # Increment the input_key to trigger updates
        st.session_state.input_key += 1

    # If there is chat history, display it in the messages container
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        for speaker, message in st.session_state.chat_history:
            # Assign a display name based on the speaker
            who = "You" if speaker == "user" else "JournaLLM"
            # Display each message in the chat container
            messages.chat_message(speaker).write(who + ': '+ str(message))

    # Provide a button to reset the chat
    if st.button('Reset Chat'):
        # Clear the chat history
        st.session_state.chat_history = []
        # Increment the input_key to trigger updates
        st.session_state.input_key += 1
        # Rerun the Streamlit app to reflect changes
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
    
    st.session_state['username'] = st.text_input("Username:")
    st.session_state['password'] = st.text_input("Password:", type="password")
    
    if st.button("Login"):
        if not st.session_state['username'] or not st.session_state['password']:
            st.error("Both username and password are required.")
        elif authenticate(st.session_state['username'], st.session_state['password']):
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
                st.success("Signup successful! You can now login.")
                #create db
            else:
                st.error("Username already exists. Please choose a different username.") 


def entry():
    st.title('JournaLLM')  # Set the title of the Streamlit app
    # Display a welcome message to the user
    st.write('Welcome to JournaLLM, your personal space for mindful reflection and goal tracking! This app is designed to help you seamlessly capture your daily thoughts, set meaningful goals, and track your progress.')
    
    # Initialize an input key in session state if it's not present, to track submissions
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    file_path = "output.txt"  # Define the path for the output file where entries will be saved

    # Create a text area in the app for users to write their journal entry
    text = st.text_area("Today's Entry")
  
    # Prepare a template with questions for the journal entry, including the current date
    template = f'''Question: What happened on {datetime.today().strftime("%B %d, %Y")}? 
    How did I feel on {datetime.today().strftime("%B %d, %Y")}? 
    What were the events that happened on {datetime.today().strftime("%B %d, %Y")}? 
    Describe your day, {datetime.today().strftime("%B %d, %Y")}. \n Answer: '''
    
    text = template + text  # Append the user's text to the template

    # Save the journal entry to a file and update the vector database when the 'Pen down' button is pressed
    if st.button('Pen down') and text:
        save_into_text_file(file_path, text)  # Call function to save the text into the specified file
        vectordb_entry()  # Call function to add the entry to the vector database
        st.write('Entry saved')  # Confirm the entry has been saved
        st.write(text)  # Display the saved text to the user
        st.session_state.input_key += 1  # Increment the session state's input key to track changes



# Main Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Login", "Signup","Entry","Advice"])

    if page == "Login":
        login_page()
    elif page == "Signup":
        signup_page()
    elif page == "Journal":
        if st.session_state.username == "":
            st.write('Please login to continue.')
        else:
            st.write(f"Logged in as {st.session_state.username}")
            entry()
    elif page == "Advice":
        if st.session_state.username == "":
            st.write('Please login to continue.')
        else:
            st.write(f"Logged in as {st.session_state.username}")
            journal()

if __name__ == "__main__":
    main()

