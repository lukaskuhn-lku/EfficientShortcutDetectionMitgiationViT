   
from tqdm import tqdm
import os

def caption_images(api, cap_mod, patch_files, n_cluster, model_name):
    cluster_descriptions = []

    for cluster in range(n_cluster):
        descriptions = []
        for idx, image in tqdm(enumerate(patch_files[cluster])):
            output = api.run(
                cap_mod[0],
                input={
                    "image": open(image, "rb"),
                    "prompt": "What is in this picture? Describe in a few words.",
                },
            )
            descriptions.append("".join(output))

        cluster_descriptions.append(descriptions)

        os.makedirs(
            f"outputs/{model_name}/descriptions/{cluster}", exist_ok=True
        )
        # write descriptions to file
        with open(
            f"outputs/{model_name}/descriptions/{cluster}/{cap_mod[1]}_descriptions.txt",
            "w",
        ) as f:
            f.write("\n".join(descriptions))
            
def summarize_cluster(replicate_api, cluster_descriptions, MODEL_NAME):
    for cluster in range(len(cluster_descriptions)):
        input = {
            "prompt": f"I extracted patches from images in my dataset where my model seems to focus on the most. I let an LLM caption these images for you. I am searching for potential shortcuts in the dataset. Can you identify one or more possible shortcuts in this dataset? Describe it in one sentence (only!) and pick the most significant. No other explanations are needed. If there isn't any shortcut obvious to you then just say so. Descriptions: \n"
            + "".join(cluster_descriptions[cluster]),
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        }

        output = replicate_api.run("meta/meta-llama-3-70b-instruct", input=input)

        text = "".join(output)

        with open(
            f"outputs/{MODEL_NAME}/descriptions/{cluster}/llama70b_shortcut.txt",
            "w",
        ) as f:
            f.write(text)