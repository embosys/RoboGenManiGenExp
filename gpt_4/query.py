import openai
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key_env = os.environ.get("ROBOGEN_API_KEY_ENV")
if api_key_env:
    os.environ["OPENAI_API_KEY"] = os.environ.get(api_key_env, "")
elif "LLM_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]
elif "YUFEI_OPENAI_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["YUFEI_OPENAI_API_KEY"] # legacy key name
api_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE")
if api_base:
    openai.api_base = api_base.rstrip("/")
def query(system, user_contents, assistant_contents, model='gpt-4', save_path=None, temperature=1, debug=False):
    
    for user_content, assistant_content in zip(user_contents, assistant_contents):
        user_content = user_content.split("\n")
        assistant_content = assistant_content.split("\n")
        
        for u in user_content:
            print(u)
        print("=====================================")
        for a in assistant_content:
            print(a)
        print("=====================================")

    for u in user_contents[-1].split("\n"):
        print(u)

    if debug:
        import pdb; pdb.set_trace()
        return None

    print("=====================================")

    start = time.time()
    
    num_assistant_mes = len(assistant_contents)
    messages = []

    messages.append({"role": "system", "content": "{}".format(system)})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx]})
        messages.append({"role": "assistant", "content": assistant_contents[idx]})
    messages.append({"role": "user", "content": user_contents[-1]})

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Missing LLM API key; set LLM_API_KEY in the repository .env")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model=os.environ.get("LLM_MODEL") or model,
        messages=messages,
        temperature=temperature
    )

    result = ''
    for choice in response.choices: 
        result += choice.message.content 

    end = time.time()
    used_time = end - start

    print(result)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump({"used_time": used_time, "res": result, "system": system, "user": user_contents, "assistant": assistant_contents}, f, indent=4)

    return result
