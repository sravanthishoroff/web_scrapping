import os
import openai
import ssl
import certifi
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.agents import load_tools, initialize_agent
from langchain.llms import AzureOpenAI


import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from dotenv import load_dotenv
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

load_dotenv()


# # Insert your OpenAI API key here
# openai_key = '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'
# os.environ["OPENAI_API_KEY"] = openai_key

# # google search api key
# serpapi_key = '46ee3c2e1384d5544419d12b94bc95a4d3afb53ee91be78d7ab6835bcb95a84e'
# os.environ["SERPAPI_API_KEY"]= serpapi_key


# Azure OpenAi endpoint configuration
# config = {
#     "openai": {
#         "base": "https://et-poc-openai.openai.azure.com/",
#         "key": "7fb0f62ba5bd43c1b427e1ecb6220af2",
#         "version": "2022-12-01",
#         "type": "azure"
#     }
# }


# Taking input from config dict and defining variable for each value  
# oai_base = config['openai']['base']
# oai_key = config['openai']['key']
# oai_ver = config['openai']['version']
# oai_type = config['openai']['type']

# os.environ["OPENAI_API_TYPE"] = oai_type
# os.environ["OPENAI_API_BASE"] = oai_base
# os.environ["OPENAI_API_VERSION"] = oai_ver
# os.environ['OPENAI_API_KEY'] = oai_key

# google search api key
# serpapi_key = '46ee3c2e1384d5544419d12b94bc95a4d3afb53ee91be78d7ab6835bcb95a84e'
# os.environ["SERPAPI_API_KEY"]= serpapi_key

# openai.api_type = oai_type
# openai.api_key = oai_key
# openai.api_base = oai_base
# openai.api_version = oai_ver


def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["span"])
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,
                                                                    chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)

    # Process the first split 
    extracted_content = extract(
        schema=schema, content=splits[0].page_content
    )
    pprint.pprint(extracted_content)
    return extracted_content




schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}


# AzureOpenAPI code
# llm = AzureOpenAI(deployment_name='text-davinci-003',model_name='text-davinci-003',temperature=0,max_tokens=1000)

# openAI 
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")




def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


urls = ["https://www.wsj.com"]
extracted_content = scrape_with_playwright(urls, schema=schema)
print(extracted_content)