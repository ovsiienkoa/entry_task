from CLIP_train import CLIP_alt
from transformers import pipeline


import argparse
from PIL import Image
def predict(model_type, texts, images:list[Image], model_path:str):
    """
    Performs zero-shot image classification using either a standard CLIP model or
    a fine-tuned CLIP model with a GLiNER text encoder.

    Args:
        model_type (str): The model variant to use: 'base' for standard CLIP pipeline,
                          or 'gliner' for the custom fine-tuned CLIP_alt model.
        texts (List[str]): A list of candidate text labels/captions to classify the image against.
        images (List[Image.Image]): A list of input images as PIL Image objects.
        model_path (str): Directory path where the custom 'gliner' model checkpoints are saved.

    Returns:
        List[str]: A list of predicted labels (text strings) for each input image.

    Raises:
        NotImplementedError: If the specified model_type is not 'base' or 'gliner'.
    """
    if model_type =='base':
        pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
        output = pipe(
            images,
            candidate_labels=texts,
        )
        # The pipeline output is a list of lists of dictionaries. Extract the top predicted label.
        true_output = [prediction[0]["label"] for prediction in output]
    elif model_type =='gliner':
        model = CLIP_alt()
        model.load(model_path,"120")
        tokens = model.preprocess(texts, images)
        output_ids = model.predict(tokens)
        # Map the predicted text index back to the actual text string label
        true_output = [texts[output_id] for output_id in output_ids]
    else:
        raise NotImplementedError

    return true_output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",
                     default="gliner",
                     help="base is CLIP from transformers, gliner is CLIP if you already trained model on your hardware ;) because it's too large and I don't want to pollute hugginface")

    parser.add_argument(
        "--labels",
        nargs='+',
        type=str,
        default=["white-horse", "black-horse", "duck"],
        help="labels with spaces"
    )
    parser.add_argument(
        "--images_paths",
        nargs='+',
        type=str,
        default=["data/raw-img/horse/OIP-0aMd_hJDVuMGMvxr1FTLSQHaGP.jpeg", "data/raw-img/horse/OIP-0b4F6_zQfANx6SOMUAlWwgHaFw.jpeg"], #black and white
        help="Path to the input image files for classification."
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="./models",
        help="Directory to save the trained model checkpoints."
    )
    args = parser.parse_args()
    images = [Image.open(image_path) for image_path in args.images_paths]
    for image in images:
        image.show()
    output = predict(args.model_type, args.labels, images, args.model_checkpoint_dir)
    print(output)
