import csv
import json
import random
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

from openai_inference import Assistant


def read_art_info_from_json(path):
    colors, objects, art_styles, textures = [], [], [], []

    with open(path) as json_file:
        data = json.load(json_file)
        colors = data["colors"]
        objects = data["objects"]
        art_styles = data["art styles"]
        textures = data["textures"]

    return colors, objects, art_styles, textures


def write_art_info_to_json(path, colors, objects, art_styles, textures):
    data = {
        "colors": colors,
        "objects": objects,
        "art styles": art_styles,
        "textures": textures
    }

    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_art_info_from_csv(path_list):
    art_style_keywords = []
    utterances = []

    for path in path_list:
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            curr_art_style_keywords = []
            curr_utterances = []

            for row in reader:
                art_style_keyword = row['art_style'].replace("_", " ")
                if art_style_keyword not in art_style_keywords and art_style_keyword not in curr_art_style_keywords:
                    curr_art_style_keywords.append(art_style_keyword)

                utterance = row['utterance']
                if len(utterance.split()) <= 10:
                    curr_utterances.append(utterance)

            art_style_keywords.extend(curr_art_style_keywords)
            utterances.extend(curr_utterances)

    print("number of art styles:", len(art_style_keywords))
    print("number of utterances:", len(utterances))

    return art_style_keywords, utterances


def augment_keywords(keywords, keyword_type, maximum_keywords=10, maximum_repetitions=3):
    assistant = Assistant()
    new_keywords = keywords[:]
    last_keyword = keywords[-1]
    n_repetitions = 0

    print(f"Augmenting {keyword_type} keywords...")

    while True:
        random.shuffle(new_keywords)
        instruction = 'Complete the list of ' + keyword_type + ' words, give me new words only and do not say anything else:\n' + '\n'.join(new_keywords)
        new_keyword = assistant.inference(instruction, temperature=0.8, max_tokens=3).strip().split("\n")[0]

        if new_keyword == last_keyword:
            n_repetitions += 1
            if n_repetitions >= maximum_repetitions:
                break
        else:
            n_repetitions = 0
            new_keywords.append(new_keyword)
            last_keyword = new_keyword

        if len(new_keywords) >= maximum_keywords:
            break

    return new_keywords


def combine_keywords(keywords_a, keywords_b):
    combined_keywords = []
    for keyword_a in keywords_a:
        for keyword_b in keywords_b:
            combined_keywords.append(keyword_a + ' ' + keyword_b)

    return combined_keywords


def augment_keyword_combinations(keyword_combinations, prefix_list, suffix_list):
    augmented_keyword_combinations = []
    for keyword_combination in keyword_combinations:
        for prefix in prefix_list:
            augmented_keyword_combinations.append(prefix + ' ' + keyword_combination)
        for suffix in suffix_list:
            augmented_keyword_combinations.append(keyword_combination + ' ' + suffix)

    return augmented_keyword_combinations


def generate_prompts_from_artemis(utterances, max_num_prompts=10):
    assistant = Assistant()
    prompts = []
    utterances = random.sample(utterances, min(max_num_prompts, len(utterances)))

    for utterance in tqdm(utterances, desc="Generating prompts from Artemis", ncols=80):
        instruction = "I am writing prompts for image style transfer. Give me a concise description of the artwork that emphasizes its texture, color scheme, and compositional flow based on the following utterance:\n"
        instruction += "<utterance> " + utterance + " </utterance>\n"
        instruction += "Include your description between the tags <description> and </description> and make it short and sweet. Do not include any other information."

        try:
            response = assistant.inference(instruction, temperature=0.5, max_tokens=64)
            prompt = response.split("<description>")[1].split("</description>")[0].strip()
        except Exception as e:
            prompt = ""
            print(e)
            print("Continuing...")
            continue

        if len(prompt.split()) > 16 or prompt == "":
            continue

        prompts.append(prompt)

    return prompts


def further_augment_prompts(existing_prompts, num_new_prompts=10, num_demonstrations=4):
    assistant = Assistant()
    new_prompts = []
    
    for _ in tqdm(range(num_new_prompts), desc="Further augmenting prompts", ncols=80):
        demonstrations = random.sample(existing_prompts, num_demonstrations)
        instruction = "I am writing prompts for image style transfer. Here are some prompt examples. Please provide a new prompt following the same style as the given prompts:\n"
        for demonstration in demonstrations:
            instruction += "<prompt> " + demonstration + " </prompt>\n"
        instruction += "Include your prompt between the tags <prompt> and </prompt>. Make it short and sweet. Do not include any other information."

        try:
            response = assistant.inference(instruction, temperature=0.7, max_tokens=64)
            prompt = response.split("<prompt>")[1].split("</prompt>")[0].strip()
        except Exception as e:
            prompt = ""
            print(e)
            print("Continuing...")
            continue

        if len(prompt.split()) > 16 or prompt == "":
            continue

        new_prompts.append(prompt)

    return new_prompts


def calculate_gpt2_perplexity_batch(sentences, model_name="./gpt2", batch_size=64):
    device = torch.device("mps")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    perplexities = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Calculating perplexities", ncols=80):
        batch = sentences[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        shift_labels = shift_labels.unsqueeze(-1)
        token_log_probs = log_probs.gather(2, shift_labels).squeeze(-1)

        token_log_probs = token_log_probs * shift_mask

        sum_log_probs = token_log_probs.sum(dim=1)
        num_tokens = shift_mask.sum(dim=1)

        avg_neg_log_probs = -sum_log_probs / num_tokens
        batch_perplexities = torch.exp(avg_neg_log_probs)

        perplexities.extend(batch_perplexities.tolist())

    return perplexities


def filter_prompts(prompts, max_num_final_prompts=200):
    to_be_filtered_prompts = []
    for prompt in prompts:
        if len(prompt.split()) > 8:
            continue
        if prompt in to_be_filtered_prompts:
            continue
        to_be_filtered_prompts.append(prompt)

    perplexities = calculate_gpt2_perplexity_batch(to_be_filtered_prompts)
    
    prompt_perplexity_pairs = list(zip(prompts, perplexities))
    prompt_perplexity_pairs.sort(key=lambda x: x[1])
    final_prompt_perplexity_pairs = prompt_perplexity_pairs[:max_num_final_prompts]

    final_prompts = [pair[0] for pair in final_prompt_perplexity_pairs]

    return final_prompts


def write_prompts_to_json(path, prompts):
    with open(path, 'w') as json_file:
        json.dump(prompts, json_file, indent=4)


def main():
    # manual keywords
    colors, objects, art_styles, textures = read_art_info_from_json("manual_keywords.json")

    # artemis keywords and utterances
    artemis_art_style_keywords, artemis_utterances = read_art_info_from_csv(
        ["artemis/artemis_dataset_release_v0.csv", "artemis/ola_dataset_release_v0.csv"]
    )
    art_styles.extend(artemis_art_style_keywords)

    # augment keyword lists based on GPT-3.5-Turbo
    colors = augment_keywords(colors, "color", maximum_keywords=256)
    objects = augment_keywords(objects, "object", maximum_keywords=256)
    art_styles = augment_keywords(art_styles, "art style", maximum_keywords=256)
    textures = augment_keywords(textures, "texture", maximum_keywords=256)

    # write augmented keywords to json
    write_art_info_to_json("augmented_keywords.json", colors, objects, art_styles, textures)
    keywords = colors + objects + art_styles + textures
    print(f'Number of colors: {len(colors)}\nNumber of objects: {len(objects)}\nNumber of art styles: {len(art_styles)}\nNumber of textures: {len(textures)}')

    # combine keywords
    combined_keywords = []
    combined_keywords.extend(combine_keywords(colors, objects))
    combined_keywords.extend(combine_keywords(colors, art_styles))
    combined_keywords.extend(combine_keywords(colors, textures))
    combined_keywords.extend(combine_keywords(art_styles, objects))
    combined_keywords.extend(combine_keywords(art_styles, textures))
    
    # augment keyword combinations with prefixes and suffixes
    prefix_list = [
        "style of", "a photo of", "a painting of", "a picture of", "an image of",
        "a pixelated photo of", "a blurry photo of", "a sketch of", "a black and white photo of"
    ]
    suffix_list = [
        "style", "style painting", "style photo", "style picture", "style image", "color"
    ]
    augmented_keywords = augment_keyword_combinations(keywords, prefix_list, suffix_list)
    augmented_keywords = random.sample(augmented_keywords, min(8192, len(augmented_keywords)))
    augmented_keyword_combinations = augment_keyword_combinations(combined_keywords, prefix_list, suffix_list)
    augmented_keyword_combinations = random.sample(augmented_keyword_combinations, min(8192, len(augmented_keyword_combinations)))

    # filter prompts based on perplexity calculated by GPT-2
    to_be_filtered_prompts = combined_keywords + augmented_keywords + augmented_keyword_combinations
    print("Number of prompts before filtering:", len(to_be_filtered_prompts))
    filtered_prompts = filter_prompts(to_be_filtered_prompts, max_num_final_prompts=8192)

    # generate prompts from Artemis utterances
    generated_prompts = generate_prompts_from_artemis(artemis_utterances, max_num_prompts=4096)

    # further augment prompts using GPT-3.5-Turbo based on existing prompts
    existing_prompts = keywords + filtered_prompts + generated_prompts
    augmented_prompts = further_augment_prompts(existing_prompts, num_new_prompts=4096, num_demonstrations=4)
    
    # get final prompts
    final_prompts = keywords + filtered_prompts + generated_prompts + augmented_prompts
    final_prompts = list(set(final_prompts))

    write_prompts_to_json("prompts.json", final_prompts)

    print("Number of final prompts:", len(final_prompts))
    

if __name__ == "__main__":
    main()