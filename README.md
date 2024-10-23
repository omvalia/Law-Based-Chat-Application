## Law Based Chat Application
### Project Overview
This application is designed to handle queries strictly related to legal matters, leveraging specialized law-based data as its knowledge source. It responds exclusively to legal inquiries, filtering out any non-law-related questions. The chatbot streamlines legal assistance by delivering accurate, law-based answers, making it a valuable resource for legal professionals and individuals seeking clarity on legal issues.
The application can handle a wide range of legal scenarios, from basic legal inquiries to more complex case-related questions, ensuring contextually relevant and precise responses.

### Technology Stack
- Frontend: Chainlit (for creating an intuitive and responsive interface)
- Backend: Python, integrated with a large language model for natural language understanding

### Libraries used:
- langchain_community: Used for handling embeddings, vector stores, and document loading.
- FAISS: Facilitates efficient similarity search to retrieve relevant legal data.
- HuggingFace Embeddings: Provides embeddings to match user queries with legal data.
- CTransformers: Runs the LLaMA-2 model for generating accurate responses to legal questions.

### Key Features
 - Legal Data Handling: The application uses a custom vector database built using FAISS and powered by HuggingFace embeddings. Documents are loaded and split into manageable chunks to ensure efficient retrieval.
 - Law-Based Query Answering: The chatbot uses the LLaMA-2-7b-chat large language model to provide tailored responses based on the legal data provided.
 - Context-Aware Responses: The bot responds only when sufficient context is available in the data; otherwise, it informs the user that it doesn't have enough information to answer.
 - Robust Filtering: The system ensures that only law-related queries are addressed, filtering out irrelevant questions.
 - Efficient User Interaction: The frontend, built using Chainlit, offers a smooth and user-friendly experience for legal professionals and individuals seeking legal assistance.

### Project Demo
![image](https://github.com/user-attachments/assets/8d957b1e-77ee-4990-92c3-a891c61807f7)
![image](https://github.com/user-attachments/assets/844d5e37-e5ef-4321-80d0-296a10cc3d03)
![image](https://github.com/user-attachments/assets/d651c8f0-3997-460b-8651-1ef5cd2482e0)
![image](https://github.com/user-attachments/assets/6501d6df-c751-4dc0-8b71-c6cf1048dd26)
![image](https://github.com/user-attachments/assets/c0ac9d64-988f-4cab-b395-98cc95cad19f)

