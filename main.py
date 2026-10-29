# import os
# import streamlit as st
# import pickle
# import time
# # from langchain import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially openai api key)

# st.title("RockyBot: News Research Tool 📈")
# st.sidebar.title("News Article URLs")

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")
# file_path = "faiss_store_openai.pkl"

# main_placeholder = st.empty()
# # llm = OpenAI(temperature=0.9, max_tokens=500)
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=500)

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)
#     # create embeddings and save it to FAISS index
#     embeddings = OpenAIEmbeddings()
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore_openai, f)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")
#             st.write(result["answer"])

#             # Display sources, if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")
#                 sources_list = sources.split("\n")  # Split the sources by newline
#                 for source in sources_list:
#                     st.write(source)




import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

load_dotenv()  # Reads GEMINI_API_KEY from .env

CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
INDEX_DIR = "faiss_store_gemini"

# Automatically picks up GEMINI_API_KEY from the environment
client = genai.Client()


class GeminiEmbeddings(Embeddings):
    """Adapts Google GenAI embedding endpoints for LangChain's FAISS."""

    def _embed(self, texts, task_type):
        vectors = []
        # Batch requests to stay safely within rate limits
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend([e.values for e in response.embeddings])
        return vectors

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

status_placeholder = st.empty()
embeddings = GeminiEmbeddings()

if process_url_clicked:
    cleaned_urls = [u.strip() for u in urls if u.strip()]
    if not cleaned_urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        try:
            status_placeholder.info("Data Loading... Started... ⏳")
            loader = UnstructuredURLLoader(urls=cleaned_urls)
            data = loader.load()

            if not data:
                st.sidebar.error("Could not extract readable text from the provided URLs.")
            else:
                status_placeholder.info("Splitting text into chunks... ✂️")
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", ","],
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                docs = splitter.split_documents(data)

                status_placeholder.info("Generating embeddings and building FAISS index... ⚙️")
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(INDEX_DIR)
                status_placeholder.success("Vector store built successfully! Ask your question below. ✅")

        except Exception as e:
            status_placeholder.empty()
            st.error(f"Error while processing URLs: {e}")

query = st.text_input("Question:")

if query:
    if not os.path.exists(INDEX_DIR):
        st.warning("Please process at least one URL before querying.")
    else:
        try:
            # allow_dangerous_deserialization is required for local pickle loading in LangChain
            vectorstore = FAISS.load_local(
                INDEX_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            docs = vectorstore.similarity_search(query, k=4)

            context = "\n\n".join(
                f"\n{d.page_content}"
                for d in docs
            )

            prompt = (
                "Answer the question using ONLY the article excerpts below. "
                "If the answer cannot be found in the context, explicitly state that you could not find it.\n\n"
                f"Excerpts:\n{context}\n\nQuestion: {query}"
            )

            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )

            st.header("Answer")
            st.write(response.text)

            sources = list(dict.fromkeys(d.metadata.get("source") for d in docs if d.metadata.get("source")))
            if sources:
                st.subheader("Sources:")
                for s in sources:
                    st.write(s)

        except Exception as e:
            st.error(f"Error while generating answer: {e}")