import sys
import rouge
from fast_bleu import BLEU, SelfBLEU
from statistics import mean
import math
import collections
import torch
import random
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead



def get_separator(line):
  if "_i_" in line:
    answer = "_i_"
  elif "_b_" in line:
    answer = "_b_"
  elif "_c_" in line:
    answer = "_c_"
  elif "_b1_" in line:
    answer = "_b1_"
  elif "_b2_" in line:
    answer = "_b2_"
  elif "_b3_" in line:
    answer = "_b3_"
  elif "_c2_" in line:
    answer = "_c2_"
  elif "_i2_" in line:
    answer = "_i2_"
  elif "_b4_" in line:
    answer = "_b4_"
  elif "_b5_" in line:
    answer = "_b5_"
  elif "_b6_" in line:
    answer = "_b6_"
  elif "_b7_" in line:
    answer = "_b7_"
  elif "_b8_" in line:
    answer = "_b8_"
  elif "_b9_" in line:
    answer = "_b9_"
  elif "_b10_" in line:
    answer = "_b10_"
  elif "_b12_" in line:
    answer = "_b12_"
  elif "_b13_" in line:
    answer = "_b13_"
  elif "_b11_" in line:
    answer = "_b11_"
  elif "_i0_" in line:
    answer = "_i0_"
  elif "_i1_" in line:
    answer = "_i1_"
  elif "_c1_" in line:
    answer = "_c1_"
  else:
      answer = "ERROR"
  return answer
 
def rouge_average_scores(hyps, refs,maxlen=110, stop_words=[]):
    rouge_scorer = rouge.Rouge()
    averaged_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
    print(len(hyps),len(refs))
    scores = rouge_scorer.get_scores(hyps, refs)
    for metric in averaged_scores.keys():
        for values in scores:
            for sub_metric in averaged_scores[metric]:
                averaged_scores[metric][sub_metric] += values[metric][sub_metric]
    for key in averaged_scores.keys():
        for sub_key in averaged_scores[key].keys():
            # IMPORTANT : In ROUGE we divide by the length of the human references.
            # F1 = 2 * (BLEU * ROUGE) / (BLEU + ROUGE)
            averaged_scores[key][sub_key] /= len(hyps)
    return averaged_scores

def split_evaluation(read_file):
    f = open(read_file, 'r')
    lines = f.readlines()
    all_answers = list()
    all_generations = list()
    for line in lines:
        print(line.split("\t"))
        if len(line)>2:
            answer = line.split("\t")[1]
            generation = line.split("\t")[3]
            a_del = get_separator(answer)
            answer = answer.replace(a_del,"").strip()
            all_answers.append(answer.replace("_end_","").strip())
            all_generations.append(generation)
    return all_answers,all_generations

def Convert(string):
    li = list(string.split(" "))
    return li

def prepare_bleu(all_generation):
    ready = list()
    for g in all_generation:
        ready.append(Convert(g))
    return ready
def self_bleu_scores(all_generation):
    #print(all_generation,len(all_generation))
    all_generation = prepare_bleu(all_generation)
    #print(all_generation, len(all_generation))

    self_bleu = SelfBLEU(all_generation)
    #print(self_bleu.get_score().values())
    for st,vals in self_bleu.get_score().items():
        print("Average SELF-BLEU for {} is {}".format(st,mean(vals)))


def len_text(all_generated):
    result = list()
    for g in all_generated:
        result.append(len(g.split(" ")))
    print("Avg Length : ", sum(result)/len(result))

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    tensor_input = tensor_input.to(device=device)
    loss=model(tensor_input, labels=tensor_input)
    print(loss,type(loss))
    loss=loss.long().cpu()
    return math.exp(loss)

def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class Template:
        pass

    obj = Template()
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
    return obj
       
def calculatePerplexity(sentence,model,tokenizer):
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) 
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return math.exp(loss)

def load_model(args):
  """Creates a model and loads in weights for it."""
  config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=None)

  model = AutoModelWithLMHead.from_pretrained(
      args.model_name_or_path,
      from_tf=bool(".ckpt" in args.model_name_or_path),
      config=config,
      cache_dir=None
  )
  
  model.to(args.device)
  return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    file_name = str(sys.argv[1])
    CHECKPOINT_PATH = str(sys.argv[2])
    print(file_name)
    all_answers,all_generation = split_evaluation(file_name)
    for a, g in zip(all_answers,all_generation):
        print(a,"\t",g)
    print(len(all_generation),len(all_answers))
    avg_scores = rouge_average_scores(all_generation,all_answers)
    print(avg_scores)
    self_bleu_scores(all_generation)
    print("FOR THE ORIGINAL TEXT")
    self_bleu_scores((all_answers))
    
    args = collections.defaultdict(
	    model_name_or_path=CHECKPOINT_PATH,
	    output_dir=CHECKPOINT_PATH,
	    n_gpu=n_gpu,
	    device=device,
	    seed = random.randint(0, 100),
	    model_type='gpt2',
	    stop_token="_end_", # Set this if your dataset has a special word that indicates the end of a text.
	    temperature=1.0,  # temperature sampling. Set this to temperature=1.0 to not use temperature.
	    k=0,  # k for top-k sampling. Set this to k=0 to not use top-k.
	    p=0.9,  # p for nucleus sampling. Set this to p=1.0 to not use nucleus sampling.
	    repetition_penalty=2,
	    length=128,  # Number of tokens to generate.
	    num_return_sequences=1,  # Number of independently computed samples to generate.
	    )
    args = dict2obj(args)
    model = load_model(args)
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=None)
    all_p = list()
    for i in all_generation:
        j=calculatePerplexity(i,model,tokenizer)
        all_p.append(j)
    print("Average Perplexity:",sum(all_p)/len(all_p))
    print("For Generated")
    len_text(all_generation)
    print("For Original")
    len_text((all_answers))

