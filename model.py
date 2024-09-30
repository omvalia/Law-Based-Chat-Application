from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore'

# Updated prompt template for legal assistant
custom_prompt_template = """You are a legal assistant. Provide helpful answers to law-based questions. 
If the question is not legal, just say 'Ask law-based questions'.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=False,  # Not returning the source docs
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Loading the model
def load_llm():
    llm = CTransformers(
        model="C:/Users/omvalia/Documents/chat_app_test/model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.1
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function with a check to ensure no duplicate responses
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})

    # Ensure the answer is clean and displayed only once
    final_answer = response['result'].strip()

    # Handling duplicates or multiple responses for non-legal questions
    if final_answer.count('Ask law-based questions') > 0:
        final_answer = 'Ask law-based questions'

    return final_answer

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Legal Assistant Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    # Call the LLM and get the result
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"].strip()

    # Ensure the answer is clean and displayed only once
    if answer == 'Ask law-based questions':
        await cl.Message(content='Ask law-based questions').send()
    else:
        await cl.Message(content=answer).send()


# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["result"]

#     # Ensure normal answers or fallback are returned only once
#     if answer.count('Ask law-based questions') > 0:
#         answer = 'Ask law-based questions'

#     # Avoid sending the same message twice
#     await cl.Message(content=answer).send()
