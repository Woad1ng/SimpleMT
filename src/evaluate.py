from easse.bleu import corpus_bleu
from easse.sari import get_corpus_sari_operation_scores, corpus_sari
from easse.fkgl import corpus_fkgl
from easse.bertscore import corpus_bertscore
from bart_score import BARTScorer
from nltk.translate import meteor_score
import json
from comet import download_model, load_from_checkpoint  # 导入COMET模块
import evaluate
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate simplification outputs with multiple metrics.")
    parser.add_argument('--input_file', type=str, default='dataset/asset/asset_orig.json', help='Input JSON file')
    parser.add_argument('--output_field', type=str, required=True, help='Field name for model output in JSON')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for BARTScore/COMET')
    parser.add_argument('--bart_checkpoint', type=str, default='facebook/bart-large-cnn', help='BARTScore checkpoint')
    parser.add_argument('--comet_model', type=str, default='Unbabel/wmt22-comet-da', help='COMET model name')
    args = parser.parse_args()

    meteor = evaluate.load('./metrics/meteor')
    Bleu = evaluate.load("./metrics/sacrebleu")
    sari = evaluate.load("./metrics/sari")
    print(args.output_field)
    # 加载数据
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    srcs = [entry[args.output_field] for entry in data]
    tgts = [[entry[f'simplify{i}'] for i in range(0, 10)] for entry in data]
    tgts_t = [[entry[f'simplify{i}'] for entry in data] for i in range(0, 10)]
    sari_sources = [entry["English"] for entry in data]
    sari_score = sari.compute(sources=sari_sources, predictions=srcs, references=tgts)
    print(f"sari: {sari_score['sari']:.4f}")
    results = Bleu.compute(predictions=srcs, references=tgts)
    print(f"BLEU: {results['score']:.4f}")

    try:
        bleu_score = corpus_bleu(sys_sents=srcs, refs_sents=tgts_t, tokenizer="13a")
        print(f"BLEU: {bleu_score:.4f}")
    except Exception as e:
        print(f"BLEU 计算出错: {e}")
    try:
        fkgl_score = corpus_fkgl(sentences=srcs, tokenizer="13a")
        print(f"FKGL: {fkgl_score:.4f}")
    except Exception as e:
        print(f"FKGL 计算出错: {e}")
    try:
        meteor_results = meteor.compute(predictions=srcs, references=tgts)
        print(f"METEOR: {meteor_results['meteor']:.4f}")
    except Exception as e:
        print(f"METEOR 计算出错: {e}")
    try:
        comet_model_path = download_model(args.comet_model)
        comet_model = load_from_checkpoint(comet_model_path)
        comet_data = [{"src": entry['English'], "mt": entry[args.output_field], "ref": entry['simplify1']} for entry in data]
        model_output = comet_model.predict(comet_data, batch_size=8, gpus=1 if 'cuda' in args.device else 0)
        print(model_output.system_score)
        print(f"COMET Score: {model_output.system_score:.4f}")
    except Exception as e:
        print(f"COMET 计算出错: {e}")
    try:
        bart_scorer = BARTScorer(device=args.device, checkpoint=args.bart_checkpoint)
        bart_score = bart_scorer.multi_ref_score(srcs, tgts, agg="max", batch_size=4)
        bart_score = sum(bart_score) / len(bart_score)
        print(f"BARTScore: {bart_score:.4f}")
    except Exception as e:
        print(f"BARTScore 计算出错: {e}")
    try:
        precision, recall, f1 = corpus_bertscore(sys_sents=srcs, refs_sents=tgts_t, tokenizer="13a")
        print(f"BERTScore - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"BERTScore 计算出错: {e}")

if __name__ == '__main__':
    main()