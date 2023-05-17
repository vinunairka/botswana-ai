from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import gradio as gr
import sys
import os
import time

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    os.environ["OPENAI_API_KEY"] = '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'

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-002", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

# def chatbot(input_text):
#     index = GPTSimpleVectorIndex.load_from_disk('index.json')
#     response = index.query(input_text, response_mode="compact")
#     return response.response

# iface = gr.Interface(fn=chatbot,
#                      inputs=gr.components.Textbox(lines=7, label="Enter your text"),
#                      outputs="text",
#                      title="Custom-trained AI Chatbot")

# index = construct_index("docs")
# iface.launch(share=True)

with gr.Blocks(title="Botswana.AI") as chatai:    
    gr.Label(label="",value="Botswana.AI")
    chatbot = gr.Chatbot(label="Messages")
    chat_message = gr.Textbox(label="Enter your text")
    clear = gr.Button("Clear")

    def respond(input_text, chat_history):
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        bot_message = index.query(input_text, response_mode="compact")
        chat_history.append((input_text, bot_message.response))
        time.sleep(1)
        return "", chat_history

    chat_message.submit(respond, [chat_message, chatbot], [chat_message, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

index = construct_index("docs")
chatai.launch(share=True)