from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from AOA import utils
import spacy
import math
from wordfreq import zipf_frequency
import argparse
import os
def main():

    dataset_dir = args.dataset_dir
    output_path = args.output_path if args.output_path else dataset_dir
    Num = args.num_beams
    logits_processor.clear()
    if args.logits_processor == 'wordfreq':
        logits_processor.append(WordFreqLogitsProcessor(tokenizer, weight=args.weight, top_k=args.top_k,
                                                        threshold_value=args.threshold_value))
    elif args.logits_processor == 'aoa':
        logits_processor.append(
            AOALogitsProcessor(tokenizer, weight=args.weight, top_k=args.top_k, threshold_value=args.threshold_value))
    with open(dataset_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)
    translated_lines = []
    model_name = os.path.basename(os.path.normpath(model_dir))
    dataset_name = os.path.splitext(os.path.basename(dataset_dir))[0]
    processor_type = args.logits_processor
    output_field = f'{model_name}_{dataset_name}_{processor_type}'
    for line in data:
        print(f"正在生成的语句：{line['Chinese']}")
        text_inputs = tokenizer(text=line['Chinese'], src_lang=args.src_lang, return_tensors="pt")
        text_inputs = text_inputs.to(device)
        output_tokens = model.generate(**text_inputs,
                                       tgt_lang=args.tgt_lang,
                                       generate_speech=False,
                                       num_beams=Num,
                                       early_stopping=True,
                                       num_return_sequences=Num,
                                       do_sample=False,
                                       logits_processor=logits_processor
                                       )
        translated_text_from_text = tokenizer.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        print(translated_text_from_text)
        line[f'3.24SM4T_decode_wf'] = translated_text_from_text
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"SM4T")
    print("\n\n\n\n")


class WordFreqLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, top_k=15, language='en',weight=1,threshold_value=5.42):

        super().__init__()
        self.tokenizer = tokenizer  # 用于解码 token 索引
        self.top_k = top_k  # 处理 logits 最高的 top_k 个 token
        self.language = language
        self.control_tokens = { 2,1}
        self.weight=weight
        self.threshold_value=threshold_value
    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # 遍历 top_k 中的每个 token，将 logit 转换为概率并乘以词频
        for batch_idx in range(logits.shape[0]):
            top_5_indices = top_k_indices[batch_idx, :5].tolist()
            if any(token_idx in self.control_tokens for token_idx in top_5_indices):
                # 如果控制 token 出现在前 5 个中，跳过调整，返回 logits

                continue
            #if zipf_frequency(self.tokenizer.decode([top_k_indices[batch_idx, 0]], skip_special_tokens=True).strip(),
            #                  self.language) > self.threshold_value:
            #    #print("continue")
            #    continue

            for i in range(self.top_k):
                token_idx = top_k_indices[batch_idx, i].item()  # 获取 token 的索引
                token_word = self.tokenizer.decode([token_idx], skip_special_tokens=True).strip()  # 解码为词汇

                # 获取该词汇的词频 (如果词频为 0，返回一个很低的概率)
                word_freq = zipf_frequency(token_word, self.language)
                word_freq = max(word_freq, 1)  # 防止词频为 0

                # 将 logit 转换为概率，然后乘以词频
                top_k_logits[batch_idx, i] = top_k_logits[batch_idx, i] + math.sqrt(word_freq) * self.weight
                #top_k_logits[batch_idx, i] = top_k_logits[batch_idx, i] + word_freq * self.weight



        logits.scatter_(-1, top_k_indices, top_k_logits)
        logits = torch.log_softmax(logits, dim=-1)
        return logits

class AOALogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, top_k=15, language='en',weight=1,threshold_value=5.42):

        super().__init__()
        self.tokenizer = tokenizer  # 用于解码 token 索引
        self.top_k = top_k  # 处理 logits 最高的 top_k 个 token
        self.language = language
        self.control_tokens = { 2,1}
        self.weight=weight
        self.threshold_value=threshold_value
    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        #print("11111111111\n\n")
        # 遍历 top_k 中的每个 token，将 logit 转换为概率并乘以词频
        for batch_idx in range(logits.shape[0]):
            top_5_indices = top_k_indices[batch_idx, :5].tolist()
            if any(token_idx in self.control_tokens for token_idx in top_5_indices):
                # 如果控制 token 出现在前 5 个中，跳过调整，返回 logits

                continue
            #if zipf_frequency(self.tokenizer.decode([top_k_indices[batch_idx, 0]], skip_special_tokens=True).strip(),
            #                  self.language) > self.threshold_value:
            #    #print("continue")
            #    continue

            for i in range(self.top_k):
                token_idx = top_k_indices[batch_idx, i].item()  # 获取 token 的索引
                token_word = self.tokenizer.decode([token_idx], skip_special_tokens=True).strip()  # 解码为词汇

                # 获取该词汇的词频 (如果词频为 0，返回一个很低的概率)
                AOA_value=utils.get_word_aoa(token_word,AOA,nlp)
                AOA_value=max(AOA_value, 1)
                #word_freq = zipf_frequency(token_word, self.language)
                #word_freq = max(word_freq, 1)  # 防止词频为 0

                # 将 logit 转换为概率，然后乘以词频
                top_k_logits[batch_idx, i] = top_k_logits[batch_idx, i] + math.sqrt(AOA_value) * self.weight
                #top_k_logits[batch_idx, i] = top_k_logits[batch_idx, i] + word_freq * self.weight



        logits.scatter_(-1, top_k_indices, top_k_logits)
        logits = torch.log_softmax(logits, dim=-1)
        return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" translation with custom logits processor")
    parser.add_argument('--model_dir', type=str, default='model/seamless-m4t-v2-large', help='Path to model directory')
    parser.add_argument('--dataset_dir', type=str, default='dataset/asset/asset_orig.json', help='Input JSON file')
    parser.add_argument('--output_path', type=str, default=None, help='Output JSON file (default: overwrite input)')
    parser.add_argument('--num_beams', type=int, default=10, help='Number of beams for generation')
    parser.add_argument('--logits_processor', type=str, default='wordfreq', choices=['wordfreq', 'aoa'], help='Which logits processor to use')
    parser.add_argument('--weight', type=float, default=0.02, help='Weight for logits processor')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k for logits processor')
    parser.add_argument('--threshold_value', type=float, default=10, help='Threshold value for logits processor')
    parser.add_argument('--src_lang', type=str, default='cmn', help='Source language code')
    parser.add_argument('--tgt_lang', type=str, default='eng', help='Target language code')
    args = parser.parse_args()

    model_dir = args.model_dir
    device = torch.device(args.device if args.device else ("cuda:1" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    logits_processor = LogitsProcessorList()
    print("加载模型完成")

    nlp = spacy.load("en_core_web_sm")
    AOA = utils.create_aoa_dict()
    main()