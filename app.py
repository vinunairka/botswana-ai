from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
from flask import Flask, render_template, request
import os

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # Change it before push to github
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

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def respond():
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        input_text = request.args.get('msg')
        bot_message = index.query(input_text, response_mode="compact")
        return bot_message.response

index = construct_index("docs")

if __name__ == "__main__":
    app.run()