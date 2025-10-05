import torch
from gliner import GLiNER
from transformers import pipeline
import argparse

def predict_labels(model_type, line, labels):
    if model_type == "ner":
        model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        try:
            labels.remove("NoF")
        except:
            pass

        predictions = model.predict_entities(line, labels, threshold=0.75)
        true_predictions = []
        for prediction in predictions:
            true_predictions.append(prediction["label"])
        if len(predictions) == 0:
            return "NoF"

    elif model_type == "nli":
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        nli_classes = []

        for animal in labels:
            labels.extend([f"there is a {animal}", f"there is no {animal}"])

        prediction = classifier(line, nli_classes)
        prediction = prediction["labels"][0]


        if prediction.startswith("there is no "):
            true_predictions = "not " + prediction[12:]
        elif prediction.startswith("there is a "):
            true_predictions = prediction[11:]

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
        default='omg, is it a horse?! I mean... maybe its a cow',
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

    print(args.labels)
    output = predict_labels(args.model_type, args.line, args.labels)
    print(output)

