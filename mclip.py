import matplotlib.pyplot as plt
import numpy as np
import transformers
import torch
import torchvision
import pytorch_lightning as pl
import open_clip
from pytorch_memlab import profile, MemReporter


# TEXT_MODEL_NAME = 'M-CLIP/LABSE-Vit-L-14'
# IMAGE_MODEL_NAME, IMAGE_PRETRAINED_NAME = 'ViT-L-14', 'laion400m_e32'
TEXT_MODEL_NAME = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
IMAGE_MODEL_NAME, IMAGE_PRETRAINED_NAME = 'ViT-B-16-plus-240', 'laion400m_e32'
Ks = [1, 5, 10]


class MCLIPConfig(transformers.PretrainedConfig):
    model_type = "M-CLIP"

    def __init__(self, modelBase='xlm-roberta-large', transformerDimSize=1024, imageDimSize=768, **kwargs):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        self.modelBase = modelBase
        super().__init__(**kwargs)


class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(
            config.modelBase,
            cache_dir=kwargs.get("cache_dir")
        )
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions,
            out_features=config.numDims
        )

    def forward(self, txt, tokenizer, device):
        txt_tok = tokenizer(txt, padding=True, return_tensors='pt').to(device)
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []


def compare_embeddings(text_embs, image_embs, logit_scale, softmax=False):
    image_features = image_embs / image_embs.norm(dim=-1, keepdim=True)
    text_features = text_embs / text_embs.norm(dim=-1, keepdim=True)

    similarities = logit_scale * text_features @ image_features.T   # 行：text、列：image
    
    if softmax:
        results = {
            'image-to-text': similarities.softmax(dim=0).cpu().detach().numpy() * 100,
            'text-to-image': similarities.T.softmax(dim=0).cpu().detach().numpy() * 100
        }
    else:
        results = {
            'image-to-text': similarities.cpu().detach().numpy(),
            'text-to-image': similarities.T.cpu().detach().numpy()
        }
        
    return results


def compare_embeddings_2(embs_1, embs_2, logit_scale):
    features_1 = embs_1 / embs_1.norm(dim=-1, keepdim=True)
    features_2 = embs_2 / embs_2.norm(dim=-1, keepdim=True)
    
    similarities = logit_scale * features_1 @ features_2.T
    
    return similarities

                         
def get_retrieval_results(logits, K, offset=0):
    results = []
    for j in range(logits.shape[1]):
        unsorted_max_indices = np.argpartition(-logits[:,j], K)[:K]
        y = logits[:,j][unsorted_max_indices]
        indices = np.argsort(-y)
        max_k_indices = unsorted_max_indices[indices]
        result = 1 if j+offset in max_k_indices else 0
        results.append(result)
    return results


def get_rSum(logits, Ks):
    recalls, rSum = {}, 0
    for label, logit in logits.items():
        for K in Ks:
            results = get_retrieval_results(logit, K)
            recall = sum(results) / len(results) * 100
            recalls[f'{label} recall@{K} [%]'] = recall
            rSum += recall
    recalls['rSum'] = rSum
    return recalls


def plot_heatmap(result_matrix, xlabel, ylabel):
    height, width = result_matrix.shape
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    im = ax.imshow(result_matrix)

    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels(["{}".format(i) for i in range(width)])
    ax.set_yticklabels(["{}".format(i) for i in range(height)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for i in range(height):
        for j in range(width):
            text = ax.text(j, i, f"{result_matrix[i, j]:.0f}",
                           ha="center", va="center", color='grey', size=20)

    fig.tight_layout()
    plt.show()


def evaluate(text_embs, image_embs, logit_scale):
    logits = compare_embeddings(text_embs, image_embs, logit_scale)
    recalls = get_rSum(logits, Ks)

    if len(text_embs) <= 20:
        fig = plt.figure()
        fig.set_size_inches(30,5)
        probs = compare_embeddings(text_embs, image_embs, logit_scale, softmax=True)
        plot_heatmap(probs['image-to-text'], xlabel='Image', ylabel='Text')
        plot_heatmap(probs['text-to-image'], xlabel='Text', ylabel='Image')
        
    return recalls


class MClipModelModule(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.text_model = MultilingualCLIP.from_pretrained(TEXT_MODEL_NAME)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

        clip_model, _, self.preprocess = \
            open_clip.create_model_and_transforms(
                model_name=IMAGE_MODEL_NAME,
                pretrained=IMAGE_PRETRAINED_NAME,
                device=self.device
            )
        for name, param in clip_model.named_parameters():
            param.requires_grad = False
        self.image_model = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()

        self.criterion1 = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion2 = torch.nn.CrossEntropyLoss().to(self.device)

    def training_step(self, batch, batch_idx):
        image_batch, caption_batch = batch
        loss, _, _ = self.calc_loss(image_batch, caption_batch)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     image_batch, caption_batch = batch
    #     loss, text_embs, image_embs = self.calc_loss(image_batch, caption_batch)
    #     self.log('validation_loss')

    #     logits = compare_embeddings(text_embs, image_embs, self.logit_scale)
    #     recalls = get_rSum(logits, Ks)
    #     self.log('validation rSum', recalls['rSum'])

    #     return loss

    def calc_loss(self, image_batch, caption_batch):
        text_embs = self.text_model(caption_batch, self.tokenizer, self.device)
        with torch.no_grad():
            pil_images = []
            for tensor_image in image_batch:
                pil_image = torchvision.transforms.functional.to_pil_image(tensor_image)
                pil_images.append(self.preprocess(pil_image))
            tensor_images = torch.stack(pil_images).to(self.device).detach()
            image_embs = self.image_model(tensor_images).detach()

        logits_1 = compare_embeddings_2(image_embs, text_embs, self.logit_scale)
        logits_2 = compare_embeddings_2(text_embs, image_embs, self.logit_scale)
        label = torch.arange(len(image_embs)).to(self.device)
        loss_1 = self.criterion1(logits_1, label)
        loss_2 = self.criterion2(logits_2, label)
        loss = (loss_1 + loss_2) / 2

        return loss, text_embs, image_embs
    
    def configure_optimizers(self):
        lr = 5e-7   # originaly 5e-5
        optimizer = torch.optim.Adam(self.text_model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)

        return optimizer


if __name__ == '__main__':
    import coco2014

    data = coco2014.DataModule(16, 16)
    model = MClipModelModule()
    reporter = MemReporter(model)
    reporter.report()
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model, data)
    reporter.report()
