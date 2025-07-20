# 🤖 RAG Chatbot

An intelligent chatbot powered by **Retrieval-Augmented Generation (RAG)** architecture, enabling natural language interaction with documents and data. It combines context-aware retrieval with modern language generation models to deliver accurate and relevant responses.

---

## 🔗 Creator

- **GitHub**: [Murugan-S-dev](https://github.com/Murugan-S-dev)  
- **LinkedIn**: [Murugan S](https://www.linkedin.com/in/murugan-s-dev2025)

---

## 🚀 Features

- 📄 Interact with PDF or document files using natural language  
- 🧠 Combines document retrieval and language model generation  
- ⚡ Fast and efficient using embeddings + vector search  
- 🌐 Simple and elegant web interface via Streamlit  

---

## 🛠️ Tech Stack

- **Programming Language**: Python  
- **Web Framework**: Streamlit  
- **LLM Backend**: HuggingFace Transformers / OpenAI GPT (customizable)  
- **Vector Store**: FAISS / ChromaDB  
- **Embeddings**: SentenceTransformers / OpenAI Embeddings  
- **Document Parsing**: PyMuPDF or similar libraries  

---

## 📦 Installation

- **Step 1**: Clone the repository  
  ```bash
  git clone https://github.com/Murugan-S-dev/RAG_CHATBOT.git


* **Step 2**: Navigate to the project directory

  ```bash
  cd RAG_CHATBOT
  ```

* **Step 3**: Install the required dependencies

  ```bash
  pip install -r requirements.txt
  ```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

> ⚙️ Make sure to configure your `.env` file with the necessary API keys and settings.

---

## 📚 How It Works

1. **Document Upload**: Users upload one or more PDF files.
2. **Text Extraction**: The app extracts and chunks text from the documents.
3. **Embedding**: Each chunk is converted into a vector embedding.
4. **Retrieval**: Relevant chunks are retrieved for a given user query.
5. **Answer Generation**: An LLM generates a contextual response based on the retrieved chunks.

---

## 🧪 Example Use Cases

* 📘 Ask questions from textbooks or academic papers
* 🧾 Summarize business documents
* ⚖️ Query legal or HR policy documents
* 🎓 Use as an educational learning assistant

---

## 👨‍💻 Author

Made with ❤️ by **Murugan S**

* GitHub: [Murugan-S-dev](https://github.com/Murugan-S-dev)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
