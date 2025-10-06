import torch
import torch.nn as nn
from datasets import load_dataset
import operator
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from gliner import GLiNER
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from transformers.modeling_outputs import BaseModelOutputWithPooling

class GLiNER_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        gliner = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)

        self.tokenizer = AutoTokenizer.from_pretrained(gliner.config.model_name)  # tokenizer for gliner
        self.seq_encoder = gliner.model.token_rep_layer.bert_layer.model
        self.text_proj_down = nn.Linear(1024, 512)

    def forward(self,
            input_ids,#(+)
            attention_mask, #(+)
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.zeros_like(input_ids)
        seq_encoded = self.seq_encoder(
            input_ids=input_ids, #idk types yet
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooler_output = seq_encoded.last_hidden_state[:, 0] #in original implementaion they take the last token, while deberta takes the first one (because biderectional)
        pooler_output = self.text_proj_down(pooler_output)
        if not return_dict:
            return (seq_encoded.last_hidden_state, pooler_output)

        return BaseModelOutputWithPooling(
                pooler_output=pooler_output
            )

class CLIP_alt(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
        self.model.text_model = GLiNER_Encoder()


        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.text_projection.parameters():
            param.requires_grad = True

        for param in self.model.text_model.text_proj_down.parameters():
            param.requires_grad = True
        # freeze all except text_proj todo


    def preprocess(self, input_texts, input_images):
        inputs = self.processor(text=input_texts, images=input_images, return_tensors="pt", padding=True)
        true_text_tokens = self.model.text_model.tokenizer(input_texts, padding=True, return_tensors="pt")
        inputs["input_ids"] = true_text_tokens["input_ids"]
        inputs["attention_mask"] = true_text_tokens["attention_mask"]
        return inputs

    def forward(self, inputs):
        outputs = self.model(return_loss = True, return_dict = False, **inputs)
        return outputs

    def save(self, path: str, hint):
        torch.save(self.model.state_dict(), f"{path}/clip_{hint}.pt")


    def load(self, path: str, hint):
        self.model.load_state_dict(torch.load(f"{path}/clip_{hint}.pt", weights_only=True))


class CLIPDataLoader:
    gliner = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)
    processor = AutoTokenizer.from_pretrained(gliner.config.model_name)
    gliner = None

    def __init__(self, dataset, batch_size, tokenizer_fn):
        self.tokenizer = tokenizer_fn
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        tokenized_datasets = tokenized_datasets.select_columns(['input_ids', 'attention_mask', 'pixel_values'])

        self.dataloader = torch.utils.data.DataLoader(
            tokenized_datasets,
            shuffle=False,
            batch_size=batch_size, #has to be the same
        )

    def tokenize_function(self, examples):
        texts = list(map(operator.itemgetter(0), examples["sentences_raw"]))
        output = self.tokenizer(input_texts=texts, input_images=examples["image"])
        input_ids = torch.tensor(output["input_ids"]).clone().detach()
        attention_mask = torch.tensor(output["attention_mask"]).clone().detach()
        pixel_values = torch.tensor(output["pixel_values"]).clone().detach()
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values}

    @staticmethod
    def batch_postpreprocess(batch):
        in_id = torch.stack(batch["input_ids"], dim=0).permute(1, 0)
        at_ms = torch.stack(batch["attention_mask"], dim=0).permute(1, 0)

        for c, chanel in enumerate(batch["pixel_values"]): #it's 6 am and I lost my ming ;)
            for d, dim in enumerate(batch["pixel_values"][c]):
                batch["pixel_values"][c][d] = torch.stack(batch["pixel_values"][c][d], dim=0)

        for c, chanel in enumerate(batch["pixel_values"]):
            batch["pixel_values"][c] = torch.stack(batch["pixel_values"][c], dim=0)

        batch["pixel_values"] = torch.stack(batch["pixel_values"], dim=0)

        px_vl = batch["pixel_values"].permute(3, 0, 1, 2)

        return {"input_ids": in_id,
                "attention_mask": at_ms,
                "pixel_values": px_vl}


def train(train_ds:CLIPDataLoader, eval_ds:CLIPDataLoader, steps:int, model:CLIP_alt):

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30
    )
    #loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    iterator = iter(train_ds.dataloader)

    writer = SummaryWriter()
    cum_loss = 0
    for step in range(steps):
        optimizer.zero_grad()

        batch = next(iterator)

        post_batch = train_ds.batch_postpreprocess(batch)
        post_batch["input_ids"] = post_batch["input_ids"].to(device)
        post_batch["attention_mask"] = post_batch["attention_mask"].to(device)
        post_batch["pixel_values"] = post_batch["pixel_values"].to(device)
        output = model(post_batch)#(input_ids = in_id, attention_mask = at_ms, token_type_ids=px_vl)

        cum_loss += output[0].item()
        output[0].backward()
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            avg_val_loss = eval(eval_ds, model)
            writer.add_scalars(f'loss', {"train": cum_loss / 20, "eval": avg_val_loss}, step)
            cum_loss = 0

        if step % 60 == 0:
            model.save("./models", str(step))

    writer.close()

def eval(eval_ds:CLIPDataLoader, model:CLIP_alt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #eval_ds = eval_ds.dataloader
    iterator = iter(eval_ds.dataloader)
    with torch.no_grad():
        model.eval()
        avg_lost = 0
        for step in range(len(eval_ds.dataloader)):
            batch = next(iterator)
            post_batch = train_ds.batch_postpreprocess(batch)
            post_batch["input_ids"] = post_batch["input_ids"].to(device)
            post_batch["attention_mask"] = post_batch["attention_mask"].to(device)
            post_batch["pixel_values"] = post_batch["pixel_values"].to(device)
            output = model(post_batch) # (input_ids = in_id, attention_mask = at_ms, token_type_ids=px_vl)

            avg_lost += output[0].item()

    model.train()
    return avg_lost / len(eval_ds.dataloader)

if __name__ == "__main__":

    train_ds_hf = load_dataset("Multimodal-Fatima/COCO_captions_validation", split='validation').select_columns(["image", "sentences_raw"])#[:66]
    test_ds_hf = load_dataset("Multimodal-Fatima/COCO_captions_test", split='test').select_columns(["image", "sentences_raw"])#[:66]


    model = CLIP_alt()
    model.load("./models", "60")
    tokenizer = model.preprocess
    train_ds = CLIPDataLoader(train_ds_hf, 64, tokenizer)
    test_ds = CLIPDataLoader(test_ds_hf, 64, tokenizer)
    train(train_ds, test_ds, 400, model)
