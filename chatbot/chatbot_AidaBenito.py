import streamlit as st
from groq import Groq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (HumanMessagePromptTemplate,MessagesPlaceholder,ChatPromptTemplate,)
from langchain_groq import ChatGroq

from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

def main():
    """
    Esta función es el punto de entrada principal de la aplicación. Configura el cliente de Groq, 
    la interfaz de Streamlit y maneja la interacción del chat.
    """
    
    # Obtener la clave API de Groq y Pinecone
    groq_api_key = os.getenv('GROQ_KEY')  
    pinecone_api_key = os.getenv('BD_KEY')  
    pc=Pinecone(api_key=pinecone_api_key)
    
    # Conectar a Pinecone y cargar el índice
    index_name = "resume-index"
    index = pc.Index(index_name)

    # Inicializar el modelo de embeddings (usando el modelo Sentence-Transformer)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo para generar embeddings

    # El título y mensaje de bienvenida de la aplicación Streamlit
    st.title("TP - Chat CV Aída Benito")
    st.write("¡Hola! Este es un ejemplo de chatbot con memoria persistente gestionada programáticamente con Langchain, utilizando Groq y Pinecone.")

    # Agregar opciones de personalización en la barra lateral
    st.sidebar.title('Personalización')
    system_prompt = st.sidebar.text_input("Mensaje del sistema:")
    model = st.sidebar.selectbox(
        'Elige un modelo',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Longitud de la memoria conversacional:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="historial_chat", return_messages=True)

    user_question = st.text_input("Haz una pregunta:")

    # Variable de estado de la sesión
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat = []
    else:
        for message in st.session_state.historial_chat:
            memory.save_context(
                {'input': message['humano']},
                {'output': message['IA']}
            )

    # Inicializar el objeto de chat Groq con Langchain
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # Si el usuario ha hecho una pregunta
    if user_question:

        # Realizar una búsqueda en Pinecone para obtener embeddings relevantes
        query_embedding = embed_model.encode(user_question).tolist()

        search_results = index.query(
            vector=query_embedding,
            top_k=3,  # Número de resultados relevantes
            include_metadata=True
        )

        # Combinar los resultados de la búsqueda con el contexto
        pinecone_context = "\n".join([f"Resultado {i+1}: {match['metadata']['text']}" for i, match in enumerate(search_results['matches'])])

        # Construir una plantilla de mensaje de chat utilizando varios componentes
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),  # Mensaje del sistema persistente
                MessagesPlaceholder(variable_name="historial_chat"),  # Historial de chat
                HumanMessagePromptTemplate.from_template("{human_input}"),  # Entrada del usuario
                SystemMessage(content=f"Contexto adicional de Pinecone: {pinecone_context}")  # Contexto de Pinecone
            ]
        )

        # Crear una cadena de conversación utilizando el LLM (Modelo de Lenguaje) de LangChain
        conversation = LLMChain(
            llm=groq_chat,  # El objeto de chat Groq LangChain inicializado anteriormente.
            prompt=prompt,  # La plantilla de mensaje construida.
            verbose=True,   # Habilita la salida detallada
            memory=memory,  # El objeto de memoria conversacional que almacena el historial
        )

        # Generar la respuesta del chatbot
        response = conversation.predict(human_input=user_question)
        message = {'humano': user_question, 'IA': response}
        st.session_state.historial_chat.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
