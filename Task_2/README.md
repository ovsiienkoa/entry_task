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