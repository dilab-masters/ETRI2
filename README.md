# ETRI2

## 장기 기억 기반 효율적인 연관 정보 선별 기술 연구
### A method that efficiently compresses long contexts to deliver high performance and memory efficiency
#### Two main ideas of our approach
##### -	Compress the retrieved documents and deliver them to the LLM
##### *	To improve QA performance and reduce memory usage
##### -	Store compressed documents during the VectorDB construction process
##### *	To enhance temporal and spatial efficiency

### 1. Environment
#### You need to log in to Hugging Face.

#### 1.1. Create conda virtual env
```markdown
conda create -n ETRI python=3.10 -y 
conda activate ETRI
```

#### 1.2. Pre-FLMR
```markdown
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl 
git clone https://github.com/LinWeizheDragon/FLMR 
cd FLMR pip install -e . 
cd third_party/ColBERT 
pip install -e . 
pip install ujson gitpython easydict ninja datasets transformers
```

#### 1.3. HOMER
```markdown
git clone https://github.com/alinlab/HOMER 
MAX_JOBS=16 pip install flash-attn==2.3.5 --no-build-isolation 
pip install accelerate matplotlib sentencepiece tqdm protobuf
```

#### 1.4. AutoCompressor
```markdown
git clone https://github.com/princeton-nlp/AutoCompressors 
pip install packaging 
pip install ujson gitpython easydict ninja datasets transformers==4.46.0 accelerate==0.24.1 sentencepiece==0.1.99 wandb 
pip install accelerate sentencepiece tqdm protobuf 
MAX_JOBS=16 pip install flash-attn==2.3.5 --no-build-isolation 
pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/rotary
```

#### 1.5. etc
```markdown
pip install openai
pip install nltk
```

### 2. Resources
#### For HOMER, the bias files for calibrating Llama-2 models can be found here.
Download KBVQA_data from [here](https://huggingface.co/datasets/BByrneLab/RAVQAV2Data)and unzip the image folders. 

### 3. Usage
#### 3.1. Baseline.py
```python
client = OpenAI(api_key=”YOUR_OPENAI_API_KEY”)
```
##### 명령어
```markdown
python baseline.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name OKVQA_PreFLMR_ViT-G \
            --experiment_name OKVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/OKVQA/eval_image/ \
            --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
            --dataset OKVQA \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 500 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
            --compute_pseudo_recall
```
##### 결과
```markdown
========= VQA Accuracy Calculation =========
Accuracy: 29.0%
EM: 21.0%
F1-Score: 22.83%
===========================================
Inference Time: 108.58 seconds
Peak GPU Memory Usage: 7070.35 MB
Average context length: 1044.93 tokens
Done! Program exiting...
```

#### 3.2. PreFLMR_HOMER.py
```python
client = OpenAI(api_key=”YOUR_OPENAI_API_KEY”)
sys.path.append('/path/to/HOMER/src/homer/')
```
##### 명령어
```markdown
python PreFLMR_HOMER.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name OKVQA_PreFLMR_ViT-G \
            --experiment_name OKVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/OKVQA/eval_image/ \
            --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
            --dataset OKVQA \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 500 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
            --compute_pseudo_recall \
	     --homer_model_path meta-llama/Llama-2-7b-chat-hf \ 
	     --scale 2 \
	     --bias_path /path/to/7b_homer_yarn_scale_2.pt
```
##### 결과
```markdown
========= VQA Accuracy Calculation =========
Accuracy: 32.0%
EM: 22.0%
F1-Score: 23.67%
Average length of original context: 6319.33
Average length of compressed context: 594.11
===========================================
Inference Time: 662.82 seconds
Peak GPU Memory Usage: 15840.11 MB
Done! Program exiting...
```


#### 3.3. PreFLMR_AutoCompressor.py
```python
sys.path.append('/path/to/AutoCompressors/')
```
##### 명령어
```markdown
  python PreFLMR_AutoCompressor.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name OKVQA_PreFLMR_ViT-G \
            --experiment_name OKVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/OKVQA/eval_image/ \
            --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
            --dataset OKVQA \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 500 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
            --compute_pseudo_recall
```
##### 결과
```markdown
========= VQA Accuracy Calculation =========
Accuracy: 0.0%
EM: 0.0%
F1-Score: 1.74%
===========================================
Inference Time: 157.92 seconds
Peak GPU Memory Usage: 40929.08 MB
Done! Program exiting...
```

#### 3.4. Semantic_Similarity_Measure.py
```python
client = OpenAI(api_key=”YOUR_OPENAI_API_KEY”)
sys.path.append('/path/to/HOMER/src/homer/')
```
#####명령어
```markdown
  python Semantic_Similarity_Measure.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name OKVQA_PreFLMR_ViT-G \
            --experiment_name OKVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/OKVQA/eval_image/ \
            --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
            --dataset OKVQA \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 500 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
            --compute_pseudo_recall \
	     --homer_model_path meta-llama/Llama-2-7b-chat-hf \ 
	     --scale 2 \
	     --bias_path /path/to/7b_homer_yarn_scale_2.pt
```
##### 결과
```markdown
========= VQA Accuracy Calculation =========
summarization quality : {'bertscore_precision': 83.28411102294922, 'bertscore_recall': 81.51279562711716, 'bertscore_f1': 82.37774521112442}
===========================================
Done! Program exiting...
```
