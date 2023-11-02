import openai
import requests
import os

# Paths of API key, result csv, prompt and output
audiolog_key_path = '~/openai_API_key.txt'
audiolog_csv_path = '~/test_results.csv'
audiolog_prompt_path = '~/log_prompt.txt'
audiolog_output_path = '~/audiolog_chatGPT_log.txt'

# load the api key
with open(audiolog_key_path, 'r') as key_f:
    line = key_f.read()
audiolog_key = line.strip('\n')

# load the result
with open(audiolog_csv_path, 'r') as log_f:
    log_lines = log_f.read()
audiolog_contet = log_lines

# load the text prompt
with open(audiolog_prompt_path, 'r') as prompt_f:
    prompt_lines = prompt_f.read()
prompt = prompt_lines

audiolog_prompt = audiolog_contet+prompt
print(audiolog_prompt)


### Generate completions using the API
############ ChatGPT

api_url = 'https://api.openai.com/v1/chat/completions'

payload = {
    "model": "gpt-3.5-turbo",  
    "messages": [{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": audiolog_prompt}]
}

headers = {
    'Authorization': f'Bearer {audiolog_key}',
    'Content-Type': 'application/json'
}

response = requests.post(api_url, json=payload, headers=headers)

data = response.json()

if 'choices' in data:
    completion = data['choices'][0]['message']['content']
    cost = int(data['usage']['total_tokens'])*0.002/1000
    print("cost USD:", cost)
    print("ChatGPT: " + completion)
    with open(audiolog_output_path, 'w') as out_f:
        out_f.write(completion)
else:
    print("API request failed or no choices")

