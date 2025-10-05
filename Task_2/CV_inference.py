from CV_train import EfficientNet
import argparse
from PIL import Image
from torchvision import transforms

def predict_labels(img:Image, model_type:str, model_path:str):
    if model_type == "en-b0":
        model = EfficientNet()
        model.load(model_path)

        preprocessed_image = model.connector(transforms.functional.pil_to_tensor(image).unsqueeze(0))
        predicted_id = model.predict(preprocessed_image)
        predicted_id = int(predicted_id.cpu().detach())
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

    image = Image.open(args.image_path)
    output = predict_labels(image, args.model_type, args.model_path)

    image.show()
    print(output)