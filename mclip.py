import numpy as np
import transformers
import torch
import pytorch_lightning as pl
import wandb


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


class MClipModelModule(pl.LightningModule):

    def __init__(self, cfg, logit_scale):
        super().__init__()
        self.cfg = cfg
        self.logit_scale = logit_scale

        self.text_model = \
            MultilingualCLIP.from_pretrained(cfg.model.text_model_name)
        self.tokenizer = \
            transformers.AutoTokenizer.from_pretrained(cfg.model.text_model_name)
        self.criterion1 = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion2 = torch.nn.CrossEntropyLoss().to(self.device)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.calc_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('validation rSum', summary='max')

        loss, text_embs, image_embs = self.calc_loss(batch)
        self.log('validation_loss', loss)

        logits = compare_embeddings(text_embs, image_embs, self.logit_scale)
        recalls = get_rSum(logits, self.cfg.eval.Ks)
        self.log('validation rSum', recalls['rSum'])

        return loss

    def calc_loss(self, batch):
        image_embs, caption_batch = batch
        text_embs = self.text_model(caption_batch, self.tokenizer, self.device)
        logits_1 = compare_embeddings_2(image_embs, text_embs, self.logit_scale)
        logits_2 = compare_embeddings_2(text_embs, image_embs, self.logit_scale)
        label = torch.arange(len(image_embs)).to(self.device)
        loss_1 = self.criterion1(logits_1, label)
        loss_2 = self.criterion2(logits_2, label)
        loss = (loss_1 + loss_2) / 2

        return loss, text_embs, image_embs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.text_model.parameters(),
            lr=self.cfg.optimizer.lr,
            betas=self.cfg.optimizer.betas,
            eps=self.cfg.optimizer.eps,
            weight_decay=self.cfg.optimizer.weight_decay
        )

        return optimizer
