import os
import json
from openai import OpenAI

from collections import defaultdict

import numpy as np
import torch
from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from easydict import EasyDict
from PIL import Image
import random

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    LlamaTokenizer
)
from flmr import (
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
)
from flmr import index_custom_collection
from flmr import create_searcher, search_custom_collection
import openai  
import time
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')
from evaluate import load

import sys
sys.path.append('/path/to/HOMER/src/homer/')

from embedding import patch_llama_for_yarn
from modeling_llama import LlamaForCausalLM

from accelerate import infer_auto_device_map, dispatch_model

client = OpenAI(api_key=”YOUR_OPENAI_API_KEY”)

def generate_gpt4_summary(context1, context2):
    prompt1 = f"Please provide a concise summary of the following text:\n\n{context1}\n\nSummary:"
    
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt1}
        ],
        max_tokens=150,  
        temperature=0.5,
        top_p=0.9
    )

    prompt2 = f"Please provide a concise summary of the following text:\n\n{context2}\n\nSummary:"
    
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt2}
        ],
        max_tokens=150,  
        temperature=0.5, 
        top_p=0.9
    )

    return response1.choices[0].message.content.strip(), response2.choices[0].message.content.strip()

def calculate_summarization_quality(references, candidates):

    bertscore = load("bertscore")

    # BERTScore
    bert_scores = bertscore.compute(predictions=candidates, references=references, lang='en')

    # Compile all scores
    summarization_quality_scores = {
        'bertscore_precision': sum(bert_scores['precision']) / len(bert_scores['precision']) * 100,
        'bertscore_recall': sum(bert_scores['recall']) / len(bert_scores['recall']) * 100,
        'bertscore_f1': sum(bert_scores['f1']) / len(bert_scores['f1']) * 100,
    }

    return summarization_quality_scores

def apply_homer_compression(model, tokenizer, context):
    prefix = "[INST] "
    suffix = " [/INST]"
    
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(model.device)
    doc_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(model.device)[:, 1:]
    suffix_ids = tokenizer(suffix, return_tensors="pt")["input_ids"].to(model.device)[:, 1:]

    max_context_len = doc_ids.size(1) - args.gen_length
    doc_ids = doc_ids[:, :max_context_len]

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        homer_embeddings = model.create_homer_prefix(prefix_ids, doc_ids, suffix_ids)["cache"].key_cache[0]

    generated_ids = model.generate(
        input_ids=prefix_ids, 
        max_new_tokens=300, 
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9
    )

    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return decoded_text
    
def generate_gpt4_answer_with_question(question, context, caption):
    prompt = f"Context: {context}\n\nImage Caption : {caption}\n\nQuestion: {question}\n\nAnswer (Please answer in one word):"

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.9)

    return response.choices[0].message.content.strip()
    
def sample_queries(ds, num_samples=5, seed=0):
    random.seed(seed)
    indices = random.sample(range(len(ds)), num_samples)
    sampled_queries = ds.select(indices)

    # for i, sample in enumerate(sampled_queries):
    #     print(f"Sample {i+1}:")
    #     print(f"Question ID: {sample['question_id']}")
    #     print(f"Question: {sample['question']}")
    #     print(f"Answers: {sample['answers']}")
    #     print(f"Image Path: {sample['img_path']}")
    #     print("==========\n")
    return sampled_queries

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def calculate_vqa_accuracy(model, tokenizer, ds):
    num_samples = len(ds)  
    
    gpt_answer_original = []
    gpt_answer_com = []
    compression_times = []

    for index, example in tqdm(enumerate(ds), total=num_samples, desc="Processing Samples"):
        retrieved_docs = [passage["passage_content"] for passage in example["retrieved_passage"]]

        concat_context = ' '.join(retrieved_docs)
    
        # Start timing for compression
        start_time = time.time()
        compressed_context = apply_homer_compression(model, tokenizer, concat_context)
        end_time = time.time()
        
        # Calculate compression time and append
        compression_time = end_time - start_time
        compression_times.append(compression_time)
    
        gpt_answer1, gpt_answer2 = generate_gpt4_summary(concat_context, compressed_context)

        gpt_answer_original.append(gpt_answer1)
        gpt_answer_com.append(gpt_answer2)
        
    print(f"summarization quality : {calculate_summarization_quality(gpt_answer_original, gpt_answer_com)}\n")

    # Print or return average compression time
    avg_compression_time = sum(compression_times) / num_samples
    print(f"Average Compression Time: {avg_compression_time:.4f} seconds")


def index_corpus(args, custom_collection):
    # Launch indexer
    index_path = index_custom_collection(
        custom_collection=custom_collection,
        model=args.checkpoint_path,
        index_root_path=args.index_root_path,
        index_experiment_name=args.experiment_name,
        index_name=args.index_name,
        nbits=args.nbits, # number of bits in compression
        doc_maxlen=512, # maximum allowed document length
        overwrite=False, # whether to overwrite existing indices
        use_gpu=args.use_gpu, # whether to enable GPU indexing
        indexing_batch_size=args.indexing_batch_size,
        model_temp_folder="tmp",
        nranks=args.num_gpus, # number of GPUs used in indexing
    )
    return index_path

def query_index(args, ds, passage_contents, passage_ids, flmr_model: FLMRModelForRetrieval):
    # Search documents
    # initiate a searcher
    searcher = create_searcher(
        index_root_path=args.index_root_path,
        index_experiment_name=args.experiment_name,
        index_name=args.index_name,
        nbits=args.nbits, # number of bits in compression
        use_gpu=args.use_gpu, # whether to enable GPU searching
    )

    def encode_and_search_batch(batch, Ks):
        # encode queries
        input_ids = torch.LongTensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.LongTensor(batch["attention_mask"]).to("cuda")
        pixel_values = torch.FloatTensor(batch["pixel_values"]).to("cuda")
        query_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        query_embeddings = flmr_model.query(**query_input).late_interaction_output
        query_embeddings = query_embeddings.detach().cpu()

        # search
        custom_quries = {
            question_id: question for question_id, question in zip(batch["question_id"], batch["question"])
        }
        ranking = search_custom_collection(
            searcher=searcher,
            queries=custom_quries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=max(Ks), # how many documents to retrieve for each query
            remove_zero_tensors=True,  # For PreFLMR, this is needed
            centroid_search_batch_size=args.centroid_search_batch_size,
        )

        ranking_dict = ranking.todict()

        # Process ranking data and obtain recall scores
        recall_dict = defaultdict(list)
        result_dict = defaultdict(list)
        for i, (question_id, pos_ids) in enumerate(zip(batch["question_id"], batch["pos_item_ids"])):
            retrieved_docs = ranking_dict[question_id]
            retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
            retrieved_docs = [doc[0] for doc in retrieved_docs]
            retrieved_doc_texts = [passage_contents[doc_idx] for doc_idx in retrieved_docs]
            retrieved_doc_ids = [passage_ids[doc_idx] for doc_idx in retrieved_docs]
            retrieved_doc_list = [
                {
                    "passage_id": doc_id,
                    "score": score,
                    "passage_content": retrieved_doc_texts[i],
                } for i, (doc_id, score) in enumerate(zip(retrieved_doc_ids, retrieved_doc_scores))
            ]
            result_dict["retrieved_passage"].append(retrieved_doc_list)
            
        batch.update(recall_dict)
        batch.update(result_dict)
        return batch

    flmr_model = flmr_model.to("cuda")
    Ks = args.Ks
    ds = ds.map(
        encode_and_search_batch,
        fn_kwargs={"Ks": Ks},
        batched=True,
        batch_size=args.query_batch_size,
        load_from_cache_file=False,
        new_fingerprint="avoid_cache",
    )

    return ds
   
def main(args):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    from datasets import load_dataset, load_from_disk
    from datasets import DatasetDict

    if args.local_data_hf == "":
        ds = load_dataset(args.dataset_hf_path, args.dataset + "_data")
    else:
        ds = DatasetDict.load_from_disk(args.local_data_hf)
    passage_ds = load_dataset(args.dataset_hf_path, args.dataset + "_passages")

    def add_path_prefix_in_img_path(example, prefix):
        if example["img_path"] != None:
            example["img_path"] = os.path.join(prefix, example["img_path"])
        return example

    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": args.image_root_dir})

    use_split = args.use_split
    ds = ds[use_split]
    passage_ds = passage_ds[f"{use_split}_passages"]
    
    ds = sample_queries(ds, num_samples=100, seed=42)

    print("========= Data Summary =========")
    print("Number of examples:", len(ds))
    print("Number of passages:", len(passage_ds))

    print("========= Indexing =========")
    # Run indexing on passages
    passage_contents = passage_ds["passage_content"]
    passage_ids = passage_ds["passage_id"]
    # passage_contents =['<BOK> ' + passage + ' <EOK>' for passage in passage_contents]

    if args.run_indexing:
        ## Call ColBERT indexing to index passages
        index_corpus(args, passage_contents)
    else:
        print("args.run_indexing is False, skipping indexing...")

    print("========= Loading pretrained model =========")
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(args.checkpoint_path, subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        args.checkpoint_path, subfolder="context_tokenizer"
    )

    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        args.checkpoint_path, subfolder="context_tokenizer"
    )

    flmr_model = FLMRModelForRetrieval.from_pretrained(
        args.checkpoint_path,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
    )
    
    image_processor = AutoImageProcessor.from_pretrained(args.image_processor_name)

    print("========= Preparing query input =========")
    
    def prepare_inputs(sample):
        sample = EasyDict(sample)

        module = EasyDict(
            {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
        )

        instruction = sample.instruction.strip()
        if instruction[-1] != ":":
            instruction = instruction + ":"
        #random_instruction = random.choice(instructions)
        text_sequence = " ".join(
            [instruction]
            + [module.separation_tokens.start]
            + [sample.question]
            + [module.separation_tokens.end]
        )

        sample["text_sequence"] = text_sequence

        return sample

    ds = ds.map(prepare_inputs)

    def tokenize_inputs(examples, query_tokenizer, image_processor):
        encoding = query_tokenizer(examples["text_sequence"])
        examples["input_ids"] = encoding["input_ids"]
        examples["attention_mask"] = encoding["attention_mask"]

        pixel_values = []
        for img_path in examples["img_path"]:
            if img_path is None:
                image = Image.new("RGB", (336, 336), color='black')
            else:
                image = Image.open(img_path).convert("RGB")
            
            encoded = image_processor(image, return_tensors="pt")
            pixel_values.append(encoded.pixel_values)

        pixel_values = torch.stack(pixel_values, dim=0)
        examples["pixel_values"] = pixel_values
        
        return examples

    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=8,
        num_proc=16,
    )

    print("========= Querying =========")
    ds = query_index(args, ds, passage_contents, passage_ids, flmr_model)
    
    print("========= VQA Accuracy Calculation =========")
    tokenizer = LlamaTokenizer.from_pretrained(args.homer_model_path)
    max_position_id = args.max_position_embeddings * args.scale
    homer_args = {
            "max_chunk_len": max_position_id // 2,
            "max_initial_chunk_len": args.max_initial_chunk_len,
            "reduction_mode": "power_max_last_calibrated",
            "layers_warmup": args.layer_warmup,
            "target_len": max_position_id // 2,
            "bias_path": args.bias_path,
    }
    model = LlamaForCausalLM.from_pretrained(
        args.homer_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        homer_args=homer_args,
    )
    calculate_vqa_accuracy(model, tokenizer, ds)
    print("===========================================")
    print("Done! Program exiting...")

if __name__ == "__main__":
    # Initialize arg parser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    # all hardcode parameters should be here
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--num_ROIs", type=int, default=9)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--dataset_hf_path", type=str, default="")
    parser.add_argument(
        "--dataset", type=str, default="EVQA"
    )
    parser.add_argument("--image_root_dir", type=str, default="./ok-vqa/")
    parser.add_argument("--use_split", type=str, default="test")
    parser.add_argument("--index_root_path", type=str, default=".")
    parser.add_argument("--index_name", type=str, default="OKVQA_GS")
    parser.add_argument("--experiment_name", type=str, default="OKVQA_GS")
    parser.add_argument("--indexing_batch_size", type=int, default=64)
    parser.add_argument("--image_processor_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--Ks", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--checkpoint_path", type=str, default="./converted_flmr")
    parser.add_argument("--run_indexing", action="store_true")
    parser.add_argument("--centroid_search_batch_size", type=int, default=None)
    parser.add_argument("--save_report_path", type=str, default=".")
    parser.add_argument("--compute_pseudo_recall", action="store_true")
    parser.add_argument("--local_data_hf", type=str, default="")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--homer_model_path", type=str, default="")
    parser.add_argument("--max_position_embeddings", type=int, default=1024)
    parser.add_argument("--gen_length", type=int, default=20)
    parser.add_argument("--max_initial_chunk_len", type=int, default=-1)
    parser.add_argument("--layer_warmup", type=int, default=12)
    parser.add_argument("--bias_path", type=str, default=None)
    parser.add_argument("--num_test_samples", type=int, default=-1)
    args = parser.parse_args()
    main(args)
