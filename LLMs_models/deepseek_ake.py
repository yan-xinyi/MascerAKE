import os
import json
import time
import requests
import re
import signal
import atexit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration  https://api.siliconflow.cn/v1/chat/completions
CONFIG = {
    "api_key": "...",
    "api_endpoint": "https://api.siliconflow.cn/v1/chat/completions",
    "model": "deepseek-ai/DeepSeek-V3",
    "max_retries": 3,
    "retry_delay": 3,
    "batch_size": 50,
    "request_timeout": 30,
    "max_workers": 5
}

HEADERS = {
    "Authorization": f"Bearer {CONFIG['api_key']}",
    "Content-Type": "application/json"
}

# File Paths
PATHS = {
    # "train": "datas/Election_Trec/train.json",
    # "test": "datas/Election_Trec/test.json",
    "train": "datas/General_Twitter/train.json",
    "test": "datas/General_Twitter/test.json",
    "output": "result/gt_deepseek_merged_mid.txt"
}


class KeyphraseExtractor:
    def __init__(self):
        self._setup_interrupt_handler()
        self.results = {}

    def _setup_interrupt_handler(self):
        atexit.register(self._save_on_exit)
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        print("\n⛔ Interrupted. Saving progress...")
        self._save_results()
        exit(0)

    def _save_on_exit(self):
        if self.results:
            self._save_results()

    def _save_results(self):
        with open(PATHS["output"], "a", encoding="utf-8") as f:
            for tweet_id, prediction in self.results.items():
                f.write(f'"{tweet_id}": "{prediction}"\n\n')
        print(f"💾 Saved {len(self.results)} predictions")
        self.results.clear()

    def _create_prompt(self, text):
        return f"""Extract 3-5 keyphrases (1-4 words each) from this text:

Text: 「{text}」

Requirements:
1. Output numbered list with quoted keyphrases
2. Only extract existing phrases (no generation)
3. Exclude URLs, hashtags without context
4. Return fewer if insufficient quality phrases

Example:
1. "election results"
2. "voter turnout"
3. "political campaign\""""

    def _call_api(self, prompt, retry_count=0):
        payload = {
            "model": CONFIG["model"],
            "messages": [
                {"role": "system", "content": "You are a precise keyphrase extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 150,
            "stop": ["\n\n"]
        }

        try:
            response = requests.post(
                CONFIG["api_endpoint"],
                headers=HEADERS,
                json=payload,
                timeout=CONFIG["request_timeout"]
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            if retry_count < CONFIG["max_retries"]:
                time.sleep(CONFIG["retry_delay"])
                return self._call_api(prompt, retry_count + 1)
            print(f"❌ Failed after retries: {str(e)}")
            return None

    def _process_response(self, response):
        if not response:
            return "API_ERROR"

        keywords = []
        for line in response.split('\n'):
            match = re.search(r'"([^"]+)"', line)
            if match and 1 <= len(match.group(1).split()) <= 4:
                keywords.append(match.group(1))

        return '\n'.join(f'"{kw}"' for kw in keywords) if keywords else "NO_VALID_PHRASES"

    def process_batch(self, batch):
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = {
                executor.submit(
                    self._process_single,
                    tweet_id,
                    text
                ): (tweet_id, text) for tweet_id, text in batch
            }

            for future in as_completed(futures):
                tweet_id, result = future.result()
                self.results[tweet_id] = result

                if len(self.results) % CONFIG["batch_size"] == 0:
                    self._save_results()
                    time.sleep(1)

    def _process_single(self, tweet_id, text):
        prompt = self._create_prompt(text)
        response = self._call_api(prompt)
        return tweet_id, self._process_response(response)


def load_data():
    with open(PATHS["train"], "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(PATHS["test"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return {**train_data, **test_data}


def prepare_text_mapping(data):
    return {
        key: " ".join(item[0] for item in values)
        for key, values in data.items()
    }


def main():
    extractor = KeyphraseExtractor()
    data = load_data()
    text_mapping = prepare_text_mapping(data)
    miss_id = [669, 2122, 7781, 8132, 12272, 13095, 30829, 31101, 34338, 35245, 36324, 39954, 42778, 48632, 49959, 51467, 55985, 57532, 64323, 69874]

    print(f"🚀 Starting processing of {len(text_mapping)} texts")

    # Process in batches
    batch = []
    for tweet_id, text in tqdm(text_mapping.items()):
        # Skip until we reach tweet_id 5925
        if int(tweet_id) not in miss_id:
            continue

        batch.append((tweet_id, text))
        if len(batch) >= CONFIG["batch_size"] * 2:  # Buffer size
            extractor.process_batch(batch)
            batch = []

    if batch:  # Process remaining
        extractor.process_batch(batch)


if __name__ == "__main__":
    main()