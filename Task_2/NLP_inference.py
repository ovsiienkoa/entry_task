import torch
from gliner import GLiNER
from transformers import pipeline
import argparse

def predict_labels(model_type, lines:list[str], labels):
    """
    Performs Natural Language Processing (NLP) inference using either Named Entity Recognition (NER)
    or Zero-Shot Natural Language Inference (NLI) to extract animal mentions or determine
    the text's assertion regarding animals.

    Args:
        model_type (str): The type of NLP task: 'ner' for entity extraction or 'nli' for zero-shot classification.
        lines (list[str]): A list of input text strings to process.
        labels (list[str]): A list of possible animal class names (including 'NoF' for NER).

    Returns:
        list: A list of predictions.
              - For 'ner': A list of lists, where each inner list contains the predicted animal labels (strings)
                for the corresponding input line. Returns ["NoF"] if no entity is found.
              - For 'nli': A list of strings, where each string is the predicted animal assertion
                (e.g., "horse" or "not cow").

    Raises:
        NotImplementedError: If the specified model_type is not 'ner' or 'nli'.
    """
    # Ensure the input is treated as a list of strings
    if isinstance(lines, str):
        lines = [lines]

    true_predictions = []
    if model_type == "ner":
        model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        try:
            labels.remove("NoF") # Remove the 'NoF' (No Found) placeholder label, as it's not a real entity class
        except:
            pass # 'NoF' was not in the list, no need to worry

        for line in lines:
            predictions = model.predict_entities(line, labels, threshold=0.75)
            true_prediction = []
            for prediction in predictions:
                true_prediction.append(prediction["label"]) # Extract the predicted label for each entity found
            if len(predictions) == 0: # If no entities were found, append "NoF" to signify no animal was mentioned
                true_prediction.append("NoF")

            true_predictions.append(true_prediction)

    elif model_type == "nli":

        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        # Prepare the candidate labels (hypotheses) for NLI
        nli_classes = []
        try:
            labels.remove("NoF") # Remove the 'NoF' (No Found) placeholder label, as it's not a real entity class
        except:
            pass # 'NoF' was not in the list, no need to worry

        # For each animal, create two hypotheses: one asserting its presence, one its absence.
        for animal in labels:
            nli_classes.extend([f"there is a {animal}", f"there is no {animal}"])
        for line in lines:
            prediction = classifier(line, nli_classes)
            prediction = prediction["labels"][0] # The highest probability label (hypothesis) is used

            if prediction.startswith("there is no "):
                true_predictions.append("not " + prediction[12:])  # Extract the animal name and prepend "not "
            elif prediction.startswith("there is a "):
                true_predictions.append(prediction[11:]) # Extract only the animal name
    else:
        raise NotImplementedError

    return true_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default='nli',
        help="nlp model type (ner / nli)"
    )
    parser.add_argument(
        "--line",
        type=str,
        default="omg, is it a horse?! I mean... maybe it's not a cow",
        help="input text, MUST BE IN DOUBLE QUOTES"
    )
    parser.add_argument(
        "--labels",
        nargs='+',
        type=str,
        default=["butterfly","cat", "chicken", "cow", "dog", "elephant", "horse", "NoF", "sheep", "spider", "squirrel"],
        help="labels with spaces"
    )
    args = parser.parse_args()

    output = predict_labels(args.model_type, args.line, args.labels)
    print(output[0]) #because we always request only 1 string

