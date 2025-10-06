Task 2. Named entity recognition + image classification
#
In this task, you will work on building your ML pipeline that consists of 2 models responsible for
totally different tasks. The main goal is to understand what the user is asking (NLP) and check if
he is correct or not (Computer Vision).

/(This approach doesn't solve the problem and I am willing to prove in the task completion.)

You will need to:
● find or collect an animal classification/detection dataset that contains at least 10 classes of animals. 

/(+/-, found only for images: https://www.kaggle.com/datasets/alessiocorrado99/animals10 & https://www.kaggle.com/datasets/muhammadabubakar691/random-images-dataset ;
And sample from COCO for training new head on the backbone from NER model;
I didn't find valuable dataset for finetuning the NER model. The only option that I consider possible is pseudo labeling (first labels will be found with regex) on huge corpuses, but the question remains still, why would ever do that if there is zero-shot ner (GLiNER) that is less than 0.3B?)

● train NER model for extracting animal titles from the text. Please use some transformer-based model (not LLM).(?, I trained new head for zro-shot NER, so in some way I did trained NER :D )

● Train the animal classification model on your dataset.(+, I linear probed Efficient Net B0 and this model is more than enough for such task. The model doesn't even require any fancy techniques like learning rate schedulers, pretraining, warm-ups, etc.)

● Build a pipeline that takes as inputs the text message and the image. ()
In general, the flow should be the following:
1. The user provides a text similar to “There is a cow in the picture.” and an image that contains any animal.
2. Your pipeline should decide if it is true or not and provide a boolean value as the output.
3. You should take care that the text input will not be the same as in the example, and the user can ask it in a different way.

=>(
Edge-cases:
* user provides a text: "there is no cow" or "The cow is absent" and the picture of a cow. NER + CV won't ever be able to correctly classify these combinations with label False (yet NLI + CV or zero-shot classification would) 
  because the only NER's task is to recognise entities
* user provides a text: "I can see a yellow monkey" and a picture of a blue monkey and now the setup of NLI + CV also fails, 
  because we can't compare embeddings directly, we have to suffer from class bottleneck
* zero shot classification is the only solution, even attempts to replace existing text encoder with new one based on NER (e.g. to save resources on the Cloud) 
  won't be as good as trained only for classification model *well, BLIP would be better but it's to close to LLMs which are forbidden to use in this task)

The solution should contain:

● Jupyter notebook with exploratory data analysis of your dataset
(I restricted the amount of objects for each class to 400, so my PC could keep up with data processing (16 gb of ram and 1660 super (6 GB of VRAM)).
There is no statistics to show, except for the fact, that in folder with 'spider' label there is a spider-monkey :D);

● Parametrized train and inference .py files for the NER model;(+)

● Parametrized train and inference .py files for the Image Classification model;(+)

● Python script for the entire pipeline that takes 2 inputs (text and image) and provides 1 boolean value as an output;
#
How to set up a project:
pip install -r requirements.txt

conda install --yes -c pytorch -c nvidia -c conda-forge --file requirements.txt

#
So, in the completion of this task:
I made linear probing of the EfficientNet-B0 for image classification;
Made Proof of Concept that NER isn't applicable in creating the system, whose goal is to determine if the caption corresponds to picture (It's a task for BLIP);
I also tested the setup with NLI that is also born dead;
I tried to create the new head for NER model to replace text embedder + text projector in CLIP, 
  so I could state that I've completed the task as you asked while paying attention to edge cases with intention of solving them without creating whole new thing 
    that wasn't described in the task.

In the end, the told fact about slow learning in CLIP became true, I simply don't have resources to use all COCO dataset, and to process 140 batches of 64 each one I spend an hour. I do believe that encoder could be retrained, but with not my budget. 
P.S. I do understand, how crucial the batch size is in Contrastive learning (overall, BYOL could take the name "VRAM - is all you need") and I am really sad, that my 6GB of VRAM could handle only 64 ents per batch :(
P.P.S. Task was actually funny, I don't know if I would ever have to do classical MNIST classification and the legendary CLIP from almost scratch (after BYOL this isn't that interesting), but now I can with all honor mark the  completion mnist classification! 
P.P.P.S. Thanks for not rejecting my application on the very first stage and I am sorry for you having to read all this thing written by some madman who hadn't slept for 40 hours.
#
Generated descriptions reviewed by human:
Description of each file:
#
CV_train:
Workflow
Initialization: The script parses command-line arguments.

Dataset Creation: An instance of CVDataset is created, which loads the images, preprocesses them, applies augmentations, and calculates the class mapping.

Model Initialization: An EfficientNet model is initialized with the class mapping provided by the dataset.

DataLoader Setup: PyTorch DataLoader objects are created for the training and evaluation/test subsets of the data.

Training: The .train() method is called to perform linear probing on the new classifier head.

Saving: The trained model and associated metadata (optimizer state, class mapping) are saved to the specified checkpoint directory.

Evaluation: The .eval_statistics() method is called to print the classification report, demonstrating the model's final performance.
#
CV_inference:
Workflow
Input: The script receives the path to an image file path and the model's checkpoint directory via command-line arguments.

Image Prep: The image is opened using PIL and converted to the standard 'RGB' format.

Prediction: The predict_labels function loads the EfficientNet model and its parameters.

Transformation: The image is transformed into a tensor and preprocessed according to the model's needs.

Output: The model performs the forward pass, and the resulting index is mapped back to the animal class name, which is then returned and printed.
#
NLP_inference:
1) NER Workflow (model_type == "ner")
This path uses a GLiNER model for zero-shot Named Entity Recognition to extract any mentioned animal names from the text.

Model Loading: The script loads the pre-trained gliner_large-v2.5 model from Hugging Face.

Label Preparation: The list of animal classes is used as the entity labels the model should search for.

Inference: The predict_entities method is called on the input text with the defined animal labels.

Result Aggregation:

If one or more animal entities are found, their corresponding labels (animal names) are collected into a list.

If no entities are found, the list contains only the placeholder label "NoF".

Output: A list of lists is returned, where each inner list contains the extracted animal labels for the corresponding input sentence.

2) NLI Workflow (model_type == "nli")
This path uses a BART-based Zero-Shot Classification pipeline to determine the most likely assertion the user is making about the animals. This is designed to handle "absence" cases (e.g., "there is no cow").

Hypothesis Generation: A set of candidate labels (hypotheses) is generated for the NLI classifier. For every animal in the class list (e.g., "cow"), two hypotheses are created: "there is a cow" and "there is no cow".

Inference: The NLI classifier compares the input text (the premise) against all generated hypotheses and returns the one with the highest probability.

Result Extraction:

If the top prediction starts with "there is no ", the script extracts the animal name and prepends "not " (e.g., "there is no cow" becomes "not cow").

If the top prediction starts with "there is a ", the script extracts only the animal name (e.g., "there is a horse" becomes "horse").

Output: A list of strings is returned, with each string representing the text's assertion (e.g., "horse" or "not cow").

3) (__main__)
Argument Parsing: Command-line arguments (model_type, line, labels) are parsed.

Inference Call: The predict_labels function is called with the arguments.

Result Display: The script prints the prediction from the returned list.
#
Full_Pipe
1) Image Classification (CV):

The input image path is used to open the image with PIL.

The CV_inference.predict_labels is called using the provided models_path.

This step returns a single string: the ground truth animal label found in the image (e.g., 'horse').

2) Text Analysis (NLP):

The input text is passed to the NLP_inference.predict_labels function, along with the specified text_model_type ('ner' or 'nli').

This step extracts the animal assertion made in the text
3) Assertion Comparison and Output: The script uses the text_model_type to determine how to compare the text assertion with the image's ground truth.
* If text_model_type is 'ner' (Named Entity Recognition):

The NER model extracts a list of all animal entities mentioned (e.g., ['cow'] or ['NoF']).

The script iterates through the extracted nlp_output list:

If any extracted animal label matches the image label, the function returns True.

If none of extracted animal label matches the image label, the function returns False.

* If text_model_type is 'nli' (Zero-Shot Natural Language Inference):
The NLI model provides a single assertion (e.g., 'cow' or 'not cow').

The script returns True only if image predicted image label corresponds to text predicted label, or the text predicted label denies entity that wasn't predicted by vision model. 

4)
Workflow 
Argument Parsing: The script requires four command-line arguments: input_text, input_image_path, text_model_type, and models_path.

Pipeline Execution: The full_predict function is called with the parsed arguments to run the entire check.
#
CLIP_train
* GLiNER_Encoder:

Loads the pre-trained gliner_large-v2.5 model and its tokenizer.

The GLiNER backbone (the BERT-like seq_encoder) is extracted to serve as the new text encoder.

A custom text_proj_down linear layer is created to project the GLiNER backbone's 1024-dimensional output down to the 512-dimensional embedding space expected by the CLIP vision encoder.

* CLIP_alt:

Loads the pre-trained clip-vit-base-patch32 model and its processor.

The original CLIP text encoder is replaced with the custom GLiNER_Encoder instance.

Weight Freezing: All parameters in the CLIP model are frozen (requires_grad = False), except for the original CLIP text_projection layer and the new GLiNER-based text_proj_down layer. This implements linear probing only on the projection heads to align the new text encoder's embeddings with the vision encoder's space.

* Data Loading and Preprocessing (CLIPDataLoader)
Dataset Loading: The script loads the COCO captions dataset from Hugging Face for multimodal training, specifically using the validation and test splits because of their sizes


The CLIP processor is used for image pre-processing (pixel_values).

The GLiNER-based tokenizer is used for text tokenization (input_ids, attention_mask).

Tokenization and Mapping:

The CLIPDataLoader initializes by applying a tokenize_function over the entire dataset using dataset.map(). 
This function extracts the first caption from the sentences_raw list and applies the custom GLiNER's preprocessing.

DataLoader Creation: A standard PyTorch DataLoader is created, wrapping the tokenized dataset.

* Training and Evaluation (train and eval functions)
Optimizer and Scheduler:

An AdamW optimizer is initialized to update only the trainable parameters (the two projection layers).

A CosineAnnealingWarmRestarts scheduler is used for learning rate management.

The contrastive loss (returned as the first element of the output tuple) is calculated and used for backpropagation and optimization.

Losses are logged to TensorBoard every 30 steps.

The model checkpoint is saved every 60 steps.

The learning rate scheduler is stepped.

Evaluation (eval):

Runs the model in torch.no_grad() mode over the evaluation dataset.

Calculates and returns the average contrastive loss on the evaluation set to monitor training progress.
#
CLIP_inference
Workflow (predict function)
* Input Gathering: The function receives the desired model_type ('base' or 'gliner'), a list of candidate possible captions (e.g., animal names), a list of input images (as PIL objects), and the model_path (if using the custom model).

If model_type is 'base' (Standard CLIP):
The script loads the standard openai/clip-vit-base-patch32 model using the Hugging Face zero-shot-image-classification pipeline.


If model_type is 'gliner' (Custom CLIP):
An instance of the custom CLIP_alt model is created.

The script loads the trained weights for the custom model from the specified model_path (using the fixed hint "120")(amount of batches that model was trained on).

Output: A list of predicted text labels (strings) is returned, with one label for each input image.

* Execution Workflow (__main__)
Argument Parsing: The script takes command-line arguments for the model_type, a space-separated list of candidate labels (text strings), a list of images_paths, and the model_checkpoint_dir.

Image Loading: The script iterates through the input image paths, opens each one using PIL, and stores them in a list.

Image Display: A loop iterates through the loaded images and calls image.show() to visually display the inputs.

Inference Call: The predict function is called to perform the zero-shot classification.

Result Display: The list of predicted labels for all input images is printed to the console.