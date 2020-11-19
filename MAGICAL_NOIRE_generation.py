def find_delimiter(original_text):
    delimiter = original_text.split("_ ")[2].split(" _")[1]
    if delimiter not in ("i","i1","i2","b","b1","b2","b3","b4","b5","b6","c","c1","c2"):
        print("ERROR")
    return "_"+delimiter+"_"

#!python Final_run_generation.py   --inputfile_val='../data_generation/data/Model_0/val_wikiplots_full_.txt'  --model_type=gpt2     --model_name_or_path='../language_model/result_model_3' --length 120 --p 0.9 --repetition_penalty 1.4     --outputfile_generated='results/M3_Generation.txt' --full_in='results/M3_Results.txt' --mtype=3


# Model Types
#type_0 = 1 	Title
#type_1 : 	Title + Outline
#type_3 : 	Title + Outline + Previous Paragraph
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

def get_model_specific_prompt(line,type_m):
    title = line.split("\t")[0]
    kw = line.split("\t")[1]
    p = line.split("\t")[2]
    if type_m==0:
        prompt_text = title
        answer_text = p
    if type_m==1:   # Title + Genre + Discourse
        prompt_text = title+"\t"+get_separator(p)
        answer_text = p
    if type_m==2:   #Title + Genre + KW + Discourse
        prompt_text = title + "\t" + kw+"\t" +get_separator(p)
        answer_text = p
    elif type_m ==3:
        out_line = line.split("_endpv_")[1]
        delimiter = get_separator(out_line)
        if delimiter == "ERROR":
            prompt_text = "ERROR"
            answer_text = "ERROR"

        prompt_text = line.split("_endpv_")[0] + " _endpv_ "+delimiter
        answer_text = out_line

    return prompt_text, answer_text

import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():

    #inputfile_val = '/content/drive/My Drive/Thesis/Data/Custom_Check.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile_val", type=str, default="")
    parser.add_argument("--outputfile_generated", type=str, default="")
    parser.add_argument("--full_in", type=str, default="")
    parser.add_argument("--mtype", type=int, default=0)




    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    #parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

###
    inputfile_val = args.inputfile_val
    outputfile_generated = args.outputfile_generated
    full_in = args.full_in
    mtype = args.mtype
    stories_saved = 0
    f = open(inputfile_val, 'r', encoding='"ISO-8859-1"')
    lines = f.readlines()
    i = 0
    all_generated = list()
    real_answers = list()
    print(" ==== STARTING GENERATION ====")
    for line in lines:
        original_text = line
        prompt_text, answer = get_model_specific_prompt(line,mtype)
        if prompt_text == "ERROR":
            print("ERROR")
            break
        real_answers.append(answer)
        print(" * Our prompt : ", prompt_text)
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )
            only_generation = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]

            generated_sequences.append(total_sequence)
            print("* Our generation",total_sequence)

            file = open(outputfile_generated, 'a+')
            #generated_sequences.append(" _end_ ")
            file.write("answer"+"\t"+answer+"\t"+"generated"+"\t"+only_generation+"\n")
            file2 = open(full_in,'a+')
            new_line = "[PROMPT]"+prompt_text+"\t"
            new_line = new_line+"[ORIGINAL]"+original_text.replace("\n","").strip()+"\t"
            new_line = new_line+"[GENERATED]"+str(generated_sequences).strip('[]')+"_end_"+"\n"
            file2.write(new_line)
            all_generated.append(generated_sequences)
            i=i+1
            if i==10:
              exit()
    return all_generated, real_answers


if __name__ == "__main__":
    main()
