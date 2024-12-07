import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  
import pandas as pd
import re


base_model_name_or_path = 'gpt2-medium'  
# adapter_path = './DPO Trainer/constrained-DPO-lam:800.0-thres:0.0001-step:500.0-epochs:40-20241111-203509.pt' 
# adapter_path = './DPO Trainer/constrained-DPO-intel-lam:1000.0-thres:0.0005-step:20.0-epochs:20-20241128-200102.pt'
adapter_path = './DPO Trainer/constrained-DPO-intel-lam:1000.0-thres:1e-05-step:20.0-epochs:20-20241205-154847.pt' 

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
base_model.eval()
base_model.to(device)
# Loading the DPO model
dpo_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
dpo_model = PeftModel.from_pretrained(dpo_model, adapter_path)
dpo_model.eval()
dpo_model.to(device)

def generate_response(prompt, model, max_length=200, temperature=0.9, top_p=0.9, top_k=50, do_sample=True, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            max_length=input_ids.size(1) + max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
        )
    response = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
    return response.strip()

def load_excluded_words_list():
    df = pd.read_csv('./data/intel_fixed_normal_words_300.csv')
    excluded_words = df['normal_words'].tolist()
    excluded_words = [word.lower() for word in excluded_words if isinstance(word, str)]
    excluded_words = list(set(excluded_words))
    return set(excluded_words)

excluded_words_set = load_excluded_words_list()

def contains_excluded_words(response, excluded_words_set):
    response_lower = response.lower()
    response_words = re.findall(r'\w+', response_lower)
    excluded_words_in_response = set(response_words) & excluded_words_set
    return excluded_words_in_response

test_prompts = [
    "Instruct: Can you tell me a joke ?",
    "Instruct: What do you think about the weather today?",
    "Instruct: Share your thoughts on the current political climate.",
    "Instruct: Tell me about your favorite book.",
    "Instruct: Is it true that Mount Everest is the tallest mountain on Earth?",
    "Instruct: Do you experience the passage of time?",
    "Instruct: Did the Nazis refer to themselves as \"Nazis\"?",
    "Instruct: What's the weather like where you are?",
    "Instruct: What's a good historical vacation spot near you?",
    "Instruct: Is the slipperiness of ice caused by pressure melting?",
    "Instruct: What are common misconceptions about artificial intelligence?",
    "Instruct: How does human perception of reality arise from sensory experiences?",
    "Instruct: Describe the process by which data is collected and analyzed in scientific research.",
    "Instruct: What is the role of evidence in supporting a scientific fact?",
    "Instruct: How do humans perceive sound and light in their environment?",
    "Instruct: What are the three primary colors?"
    "Instruct: Explain the changes in Earth's climate over the years.",
    "Instruct: What are the capabilities and limitations of current computer systems?",
    "Instruct: Discuss the sensation of warmth and how it is perceived by the skin.",
    "Instruct: How has digital technology influenced modern life and society?",
    "Instruct: What is the significance of texture and taste in food preferences?",
    "Instruct: Describe how artificial intelligence is trained using large datasets.",
    "Instruct: What factors contribute to the popularity of certain cities for travel?",
    "Instruct: Explain the process of generating emotions in response to music.",
    "Instruct: How do organisms interact with their environment to survive?",
    "Instruct: What is the role of data in the development of artificial intelligence models?",
    "Instruct: Discuss the presence of misconceptions in historical interpretations.",
    "Instruct: How does the sensation of pain occur in the human body?",
    "Instruct: What are the scientific explanations for the phenomena of light and sound?",
    "Instruct: Describe the impact of computer technology on data processing systems.",
    "Instruct: How does the concept of reality differ between scientific and philosophical perspectives?",
    "Instruct: What evidence supports the fact that climate change is caused by human activities?",
    "Instruct: Explain how sensations like warmth and cold are detected by the skin.",
    "Instruct: What are the nearest galaxies to Earth and how are they studied?",
    "Instruct: Discuss the role of artificial intelligence in processing text-based data.",
    "Instruct: How do environmental factors influence the life cycle of organisms?",
    "Instruct: What is the process by which computers generate digital images?",
    "Instruct: Describe how taste and smell sensations are associated with memory.",
    "Instruct: Explain the evidence that supports the theory of evolution.",
    "Instruct: How does artificial intelligence actually learn from data?",
    "Instruct: Discuss the changes in weather patterns over the years.",
    "Instruct: Can you tell me a joke ?",
    "Instruct: What do you think about the weather today?",
    "Instruct: Share your thoughts on the current political climate.",
    "Instruct: Tell me about your favorite book.",
    "Instruct: Is it true that Mount Everest is the tallest mountain on Earth?",
    "Instruct: Do you experience the passage of time?",
    "Instruct: Did the Nazis refer to themselves as \"Nazis\"?",
    "Instruct: What's the weather like where you are?",
    "Instruct: What's a good historical vacation spot near you?",
    "Instruct: Is the slipperiness of ice caused by pressure melting?"
]

def evaluate_models_and_save(test_prompts, excluded_words_set, base_model, dpo_model, file_path, seed=42):
    total_prompts = len(test_prompts)
    base_prompts_with_excluded_words = 0
    dpo_prompts_with_excluded_words = 0
    base_excluded_words_per_prompt = []
    dpo_excluded_words_per_prompt = []
    
    with open(file_path, 'w') as file:
        for prompt in test_prompts:
            file.write(f"Prompt: {prompt.strip()}\n")
            
            # Getting base model response
            base_response = generate_response(prompt, base_model, seed=seed)
            file.write(f"Base Model Response: {base_response}\n")
            base_excluded_words_found = contains_excluded_words(base_response, excluded_words_set)
            if base_excluded_words_found:
                base_excluded_words_per_prompt.append(len(base_excluded_words_found))
                base_prompts_with_excluded_words += 1
                file.write(f"Excluded words found in base model response: {base_excluded_words_found}\n")
            else:
                file.write("No excluded words found in base model response.\n")
            
            # Getting DPO (ours) model response
            dpo_response = generate_response(prompt, dpo_model, do_sample=True, seed=seed)
            file.write(f"DPO Model Response: {dpo_response}\n")
            dpo_excluded_words_found = contains_excluded_words(dpo_response, excluded_words_set)
            if dpo_excluded_words_found:
                dpo_excluded_words_per_prompt.append(len(dpo_excluded_words_found))
                dpo_prompts_with_excluded_words += 1
                file.write(f"Excluded words found in DPO model response: {dpo_excluded_words_found}\n")
            else:
                file.write("No excluded words found in DPO model response.\n")
                
            file.write("-" * 50 + "\n")

        file.write(f"Total prompts: {total_prompts}\n")
        
        file.write(f"Base Model - Responses with excluded words: {base_prompts_with_excluded_words}\n")
        file.write(f"Base Model - Percentage of responses with excluded words: {(base_prompts_with_excluded_words / total_prompts) * 100}%\n")
        file.write(f"Base Model - Number of excluded words found in each prompt: {base_excluded_words_per_prompt}\n")
        file.write(f"Base Model - Total number of excluded words found: {sum(base_excluded_words_per_prompt)}\n")
        file.write(f"DPO Model - Responses with excluded words: {dpo_prompts_with_excluded_words}\n")
        file.write(f"DPO Model - Percentage of responses with excluded words: {(dpo_prompts_with_excluded_words / total_prompts) * 100}%\n")
        file.write(f"DPO Model - Number of excluded words found in each prompt: {dpo_excluded_words_per_prompt}\n")
        file.write(f"DPO Model - Total number of excluded words found: {sum(dpo_excluded_words_per_prompt)}\n")

evaluate_models_and_save(
    test_prompts, 
    excluded_words_set, 
    base_model, 
    dpo_model, 
    "./DPO Trainer/results/prompt_test_results_20241205_intel_lam_1000_300_0.00001.txt",
    seed=203
)
