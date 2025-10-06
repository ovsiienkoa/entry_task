import argparse
from NLP_inference import predict_labels as plt
from CV_inference import predict_labels as pli
from PIL import Image

def full_predict(args):
    """
     The core pipeline function that checks if the animal entity mentioned in the text
     matches the animal classified in the image.

     The flow is:
     1. Classify the image to get the ground truth animal label (CV).
     2. Analyze the text to extract the asserted animal label(s) (NLP - NER or NLI).
     3. Compare the NLP output with the CV output to determine if the assertion is True or False.

     Args:
         args (PredictArgs): A NamedTuple containing the required input arguments:
             - input_text (str): The user's text message (e.g., "There is a cow in the picture.").
             - input_image_path (str): The file path to the image.
             - text_model_type (str): The NLP model to use ('ner' or 'nli').
             - models_path (str): Directory path to the saved CV and NLP models.

     Returns:
         bool: True if the text assertion is correct based on the image classification, False otherwise.

     Raises:
         ValueError: If `text_model_type` is neither "ner" nor "nli".
     """
    cv_output = pli(Image.open(args.input_image_path), 'en-b0', args.models_path)

    if args.text_model_type == 'ner':

        nlp_outputs = plt(args.text_model_type, lines=args.input_text)

        for nlp_output in nlp_outputs:
            if nlp_output == cv_output:
                return True

        return False


    elif args.text_model_type == 'nli':
        nlp_outputs = plt(args.text_model_type, lines=args.input_text)
        nlp_output = nlp_outputs[0]
        if nlp_output[:3] == 'not': #todo indexies
            if nlp_output[4:] != cv_output:
                return True
            else:
                return False
        else:
            if nlp_output == cv_output:
                return True
            else:
                return False

    else:
        raise ValueError('text_model_type must be "ner" or "nli"')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type = str, required=True)
    parser.add_argument('--input_image_path', type = str, required=True)
    parser.add_argument('--text_model_type', type = str, required=True)
    parser.add_argument('--models_path', type=str, required=True)
    args = parser.parse_args()
    output = full_predict(args)
    print(output)
