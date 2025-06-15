import json
import openai
import time
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 挂载 Google Drive
print("CUDA Available:", torch.cuda.is_available())

# 设置 OpenAI API Key
openai.api_key = "sk-..."  # 替换为你自己的密钥

# 读取原始预测文件
predict_file_path = "result/GPT_ELE_PREDICT.txt"
with open(predict_file_path, 'r', encoding='utf-8') as f:
    previous_results = json.load(f)
train_path = 'datas/Election_Trec/train.json'
test_path = 'datas/Election_Trec/test.json'

# 加载数据
train_file = json.load(open(train_path, 'r', encoding='utf-8'))
test_file = json.load(open(test_path, 'r', encoding='utf-8'))
data = {**train_file, **test_file}

# 标签映射
label_to_ids = {'none': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4, "O": 5}
ids_to_label = {v: k for k, v in label_to_ids.items()}

# 找出需要重新预测的tweet ID（错误或关键词超长）
retry_ids = []
for idx, val in previous_results.items():
    if val == "Error":
        retry_ids.append(idx)
    else:
        phrases = [line.split('"')[1] for line in val.splitlines() if '"' in line]
        if any(len(phrase.split()) > 4 for phrase in phrases):
            retry_ids.append(idx)

print(f"🔁 Need to reprocess {len(retry_ids)} tweets.")


# 构建 tweet 文本映射
id_to_text = {}
for key in data.keys():
    sentence = ' '.join([item[0] for item in data[key]])
    id_to_text[key] = sentence

# GPT Prompt模板
def create_prompt(tweet_text):
    return f'''Here is a tweet: "{tweet_text}"

Please extract 5 keyphrases (each keyphrase MUST only contain 1-4 words) that best summarize the important information or topics of the tweet.
The output should be a numbered list of keyphrases, with each keyphrase in quotation marks.
Notice: the keyphrases should only be extracted from tweet, do not generate any!
If there are fewer than 5 important keyphrases, output only the number you think is appropriate.

Example output:
1. "keyword1"
2. "keyword phrase two"
3. "another keyword"'''

# 提取数据
data_sens, data_tags = [], []
data_id_map = []
true_keyphrases_all = []
for key in data.keys():
    if key not in retry_ids:
        continue
    sentence, tags, true_phrases, current_phrase = '', [], [], []
    for item in data[key]:
        word, label = item[0], item[-1]
        sentence += word + ' '
        tags.append(label_to_ids.get(label, 0))
        if label == 'B':
            if current_phrase:
                true_phrases.append(' '.join(current_phrase))
                current_phrase = []
            current_phrase.append(word)
        elif label == 'I':
            current_phrase.append(word)
        elif label == 'E':
            current_phrase.append(word)
            true_phrases.append(' '.join(current_phrase))
            current_phrase = []
        elif label == 'S':
            if current_phrase:
                true_phrases.append(' '.join(current_phrase))
                current_phrase = []
            true_phrases.append(word)
        else:
            if current_phrase:
                true_phrases.append(' '.join(current_phrase))
                current_phrase = []
    if current_phrase:
        true_phrases.append(' '.join(current_phrase))
    data_sens.append(sentence.strip())
    data_tags.append(tags)
    data_id_map.append(key)
    true_keyphrases_all.append(true_phrases)

# GPT生成关键短语
new_results = {}
batch_counter = 0
save_path = '/content/drive/MyDrive/HPC_AKE/result/GPT_ELE_PREDICT_MID.txt'
for idx, tweet_text in enumerate(data_sens):
    tweet_id = data_id_map[idx]
    prompt = create_prompt(tweet_text)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for keyphrase extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        output_text = response['choices'][0]['message']['content'].strip()
        new_results[tweet_id] = output_text
        print(f"✅ Processed tweet {tweet_id}")
        if (idx + 1) % 10 == 0:
            print("⏸️ Pausing for 10 seconds to avoid rate limit...")
            time.sleep(10)
    except Exception as e:
        print(f"❌ Error processing tweet {tweet_id}: {e}")
        new_results[tweet_id] = "Error"
    batch_counter += 1
    if batch_counter % 100 == 0 or idx == len(retry_ids) - 1:
      with open(save_path, 'a', encoding='utf-8') as f:
          for tweet_id, prediction in new_results.items():
              f.write(f'"{tweet_id}": "{prediction}"\n\n')
      new_results.clear()  # 清空新预测缓存，防止重复写入
      print(f"💾 Autosaved {batch_counter} new predictions to {save_path}")