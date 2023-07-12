from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import openai 
import math
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import SystemMessage, HumanMessage
import asyncio
import langchain
from langchain.cache import SQLiteCache
import os.path as osp
import json
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
import os
import logging
from async_api import process_api_requests_from_file
import tiktoken
import pickle
import json
import google.generativeai as palm
from utils import print_str_to_file
import torch
import time




def persist_cache_to_disk(filename):
    def decorator(original_func):
        try:
            cache = pickle.load(open(filename, 'rb'))
        except (IOError, ValueError):
            cache = {}


        def new_func(*args, **kwargs):
            str_repr = json.dumps([args, kwargs], sort_keys=True)
            if str_repr not in cache:
                cache[str_repr] = original_func(*args, **kwargs)
                pickle.dump(cache, open(filename, "wb"))
            return cache[str_repr]

        return new_func

    return decorator

def load_yaml_file(filename = 'config.yaml'):
    with open(filename, 'r') as stream:
        data = load(stream=stream, Loader=Loader)
    return data 


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   res = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
   return res
 
def openai_ada_api(input_list, model_name = 'text-embedding-ada-002', max_len = 8190, max_batch = 1024):
    if len(input_list) < max_batch:
        input_list = [x[:max_len] for x in input_list]
        res = openai.Embedding.create(input = input_list, model=model_name)['data']
        res = [x['embedding'] for x in res]
        return res
    else:
        input_list = [x[:max_len] for x in input_list]
        total_res = []
        total_batch_num = math.ceil(len(input_list) / max_batch)
        for i in tqdm(range(total_batch_num)):
            sub_input_list = input_list[i * max_batch: (i + 1) * max_batch]
            res = openai.Embedding.create(input = sub_input_list, model=model_name)['data']
            res = [x['embedding'] for x in res]
            total_res.extend(res)
        return total_res

def openai_text_davinci_003(prompt, api_key):
    response = openai.Completion.create(
    model='text-davinci-003', 
    prompt=prompt,
    temperature=0,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    api_key=api_key
    )
    return response['choices'][0]['text']


def openai_text_api(input_text, api_key, model_name = "gpt-3.5-turbo", temperature = 0):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": input_text}],
        temperature=temperature,
        api_key=api_key)
    return response 

@persist_cache_to_disk("./ogb/res_chat.pkl")
def openai_text_api_list(input_texts):
    out = []
    for x in tqdm(input_texts):
        resp = openai_text_api(x)
        out.append(resp)
    return out

#df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
#df.to_csv('output/embedded_1k_reviews.csv', index=False)

async def openai_query_with_cost(instructions, generate_prompt):
    """
        Given a list of instructions, query the corresponding openai api, and estimate the token
        usage and cost
    """
    results = []
    total_price = 0
    with get_openai_callback() as cb:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        prompts = generate_prompt(instructions)
        tasks = [chat_generate(llm, instruction, message) for instruction, message in tqdm(prompts)]
        results = await asyncio.gather(*tasks)
        total_price = cb.total_cost
    return results, total_price


async def chat_generate(agent, instruction, message):
    if instruction:
        res = await agent([instruction, message])
    else:
        res = await agent([message])
    print("Generate")
    return res

def generate_prompt_for_ogb_arxiv(instructions):
    generate_prompts = []
    for line in instructions:
        title, abstract, category = line['title'], line['abstract'], line['category_name']
        # instruction = SystemMessage(content="")
        message = HumanMessage(content=f"Given the title and abstract of a paper from arxiv.\n Title: {title}\nAbstract: {abstract}\n Summarize the key points of this paper which can best represent its category.")
        generate_prompts.append((None, message))
    return generate_prompts

def generate_prompt_for_correct(texts, max_tokens = 768):
    generate_prompts = []
    for line in texts:
        line = line[:max_tokens]
        message = HumanMessage(content=f"Most words of the following text is misspelled, correct them \n{line}")
        generate_prompts.append((None, message))
    return generate_prompts

def generate_request_json_file_correct(texts, max_tokens = 300, filename = 'correct.jsonl'):
    filename = osp.join("./ogb/data", filename)
    jobs = [{"model": "gpt-3.5-turbo", "messages": [{'role': 'user', 'content': f"It seems a lot of words from the following paragraph lose some alphas in the end, can you help me correct them\n{line[:max_tokens]}"}]} for line in texts]
    with open(filename, "w+") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


async def call_async_api(request_filepath, save_filepath, request_url, api_key, max_request_per_minute, max_tokens_per_minute, sp, ss):
    await process_api_requests_from_file(
            requests_filepath=request_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_request_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name='cl100k_base',
            max_attempts=int(2),
            logging_level=int(logging.INFO),
            seconds_to_pause=sp,
            seconds_to_sleep=ss
        )
    

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_string(string: str, model = "text-davinci-003") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_chat_input_file(input_text, model_name = 'gpt-3.5-turbo'):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['model'] = model_name
        obj['messages'] = [
            {
                'role': 'user',
                'content': text 
            }
        ]
        jobs.append(obj)
    return jobs 


@persist_cache_to_disk("./async_req_davinci.pkl")
def generate_davinci_003_input_file(input_text, model_name = 'text-davinci-003', max_token = 4096, temperature = 0.7, log_probs = None):
    jobs = []
    for text in input_text:
        obj = {}
        obj['model'] = model_name
        obj['messages'] = [
            {
                "model": "text-davinci-003",
                "prompt": text,
                "max_tokens": max_token,
                "temperature": temperature,
                "stream": False,
                "logprobs": log_probs                            
            }
        ]
        jobs.append(obj)
    return jobs 


def load_result_from_jsonline(json_file_name):
    openai_result = []
    with open(json_file_name, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            openai_result.append(json_obj[1]['choices'][0]['message']['content'])
    return openai_result



async def async_openai_text_api(input_text, api_key, model_name = "gpt-3.5-turbo"):
    response = await openai.ChatCompletion.acreate(
        model=model_name,
        messages=[{"role": "user", "content": input_text}],
        temperature=0.7,
        api_key=api_key)
    return response['choices'][0]['message']['content']




def efficient_openai_text_api(input_text, filename, savepath, sp, ss, api_key="change_this_to_your_key", rewrite = True):
    if not osp.exists(savepath) or rewrite:
        jobs = generate_chat_input_file(input_text)
        with open(filename, "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")
        asyncio.run(
            call_async_api(
                filename, save_filepath=savepath,
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=api_key,
                max_request_per_minute=1000, 
                max_tokens_per_minute=90000,
                sp=sp,
                ss=ss
            )
        )
    openai_result = []
    with open(savepath, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            idx = json_obj[-1]
            if isinstance(idx, int):
                openai_result.append((json_obj[1]['choices'][0]['message']['content'], idx))
            else:
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'])
                openai_result.append((new_result['choices'][0]['message']['content'], idx))
    openai_result = sorted(openai_result, key=lambda x:x[-1])
    return openai_result




def google_text_generate_api(output_path, prompts, api_key = "change_this_to_your_key", model = 'models/text-bison-001', max_out_tokens = 512):
    if os.path.exists(osp.join(output_path, 'total.pt')):
        return torch.load(osp.join(output_path, 'total.pt'))
    palm.configure(api_key=api_key)
    results = []
    for i, prompt in enumerate(tqdm(prompts)):
        completion = palm.generate_text(
            model=model,
            prompt=prompt,
            temperature=0,
            # The maximum length of the response
            max_output_tokens=max_out_tokens
        )
        time.sleep(2)
        # import ipdb; ipdb.set_trace()
        results.append(completion.result)
        output_file_path = osp.join(output_path, f"{i}.txt")
        print_str_to_file(completion.result, output_file_path)
    torch.save(results, osp.join(output_path, 'total.pt'))
    return results


    
