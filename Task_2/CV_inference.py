from CV_train import EfficientNet
import argparse
from PIL import Image
from torchvision import transforms

def predict_labels(img:Image, model_type:str, model_path:str):
    """
       Performs inference on a single image using a trained CV model.

       The image is preprocessed according to the model's requirements (EfficientNet-B0 weights),
       and the predicted class label is returned.

       Args:
           img (Image): The input image as a PIL Image object.
           model_type (str): The type of model to use (e.g., 'en-b0').
           model_path (str): Directory path where the trained model checkpoints are saved.

       Returns:
           str: The predicted class label (animal name).

       Raises:
           NotImplementedError: If the specified model_type is not supported.
       """

    if model_type == "en-b0":
        model = EfficientNet()
        model.load(model_path)

        preprocessed_image = model.connector(transforms.functional.pil_to_tensor(img).unsqueeze(0)) #from image to prepa tensor
        predicted_id = model.predict(preprocessed_image)
        predicted_id = int(predicted_id.cpu().detach())

        # Create an inverse mapping from index to label string
        index_to_label = {value: key for key, value in model.label_to_index.items()}
        predicted_label = index_to_label[predicted_id]
    else:
        raise NotImplementedError

    return predicted_label
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default='en-b0',
        help="path to saved model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='./models',
        help="path to saved model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default = "data/raw-img/horse/OIP-0_k5iMO9dDq6IIyKwCQfeAHaE8.jpeg",
        help="Path to the input image file for classification."
    )
    args = parser.parse_args()

    image = Image.open(args.image_path).convert('RGB') # Ensure image is in RGB format
    output = predict_labels(image, args.model_type, args.model_path)

    image.show()
    print(output)