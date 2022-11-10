import matplotlib.pyplot as plt
import numpy as np
import open_clip
import tqdm
import matplotlib.pyplot as plt
import transformers
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
import torchvision


PATH_PREFIX = "/content/drive/MyDrive/Colab Notebooks/Multilingual-CLIP/clip_data/"
CAPTIONS_TRAIN_JP = PATH_PREFIX + 'STAIR-captions/stair_captions_v1.2_train.json'
CAPTIONS_TRAIN_EN = PATH_PREFIX + 'annotations/captions_train2014.json'
CAPTIONS_VAL_JP = PATH_PREFIX + 'STAIR-captions/stair_captions_v1.2_val.json'
CAPTIONS_VAL_EN = PATH_PREFIX + 'clip_data/annotations/captions_val2014.json'
BATCH_SIZE = 128
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
        self.transformer = transformers.AutoModel.from_pretrained(config.modelBase, cache_dir=kwargs.get("cache_dir"))
        self.LinearTransformation = torch.nn.Linear(in_features=config.transformerDimensions,
                                                    out_features=config.numDims)

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


def get_models(text_model_name, image_model_name, image_pretrained_name):
    text_model = MultilingualCLIP.from_pretrained(text_model_name)
    text_model.to(device).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)

    clip_model, _, compose = \
        open_clip.create_model_and_transforms(image_model_name, pretrained=image_pretrained_name)
    for name, param in clip_model.named_parameters():
        param.requires_grad = False
    clip_model.to('cpu').eval()
    image_model = clip_model.visual
    for name, param in image_model.named_parameters():
        param.requires_grad = True
    image_model.to(device).eval()
    logit_scale = clip_model.logit_scale.exp().float().to(device)
    
    return text_model, tokenizer, image_model, compose, logit_scale


def get_text_embs(caption_batch, text_model, tokenizer, device):
    with torch.no_grad():
        text_embs = text_model(caption_batch, tokenizer, device).to('cpu')   # avoid OOM

    return text_embs


def get_image_embs(image_batch, image_model, preprocess, device):
    pil_images = []
    for tensor_image in image_batch:
        pil_image = torchvision.transforms.functional.to_pil_image(tensor_image)
        pil_images.append(preprocess(pil_image))
    tensor_images = torch.stack(pil_images)
    with torch.no_grad():
        image_embs = image_model(tensor_images.to(device)).float()

    return image_embs


def compare_embeddings(text_embs, image_embs, logit_scale, softmax=False):
    # normalized features
    image_features = image_embs / image_embs.norm(dim=-1, keepdim=True)
    text_features = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    similarities = logit_scale * text_features @ image_features.T   # 行：text、列：image
    
    # detach
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
    # normalized features
    features_1 = embs_1 / embs_1.norm(dim=-1, keepdim=True)
    features_2 = embs_2 / embs_2.norm(dim=-1, keepdim=True)
    
    # cosine similarity as logits
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

    # Create X & Y Labels
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


def finetune(dataloader, train_subsetloader, val_subsetloader, text_model, tokenizer, image_model, preprocess, logit_scale, device, epochs):
    fetchSize = 16

    lr = 5e-7   # originaly 5e-5
    optimizer1 = torch.optim.Adam(text_model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CrossEntropyLoss()

    recalls_train, recalls_val = \
        get_recalls(train_subsetloader, val_subsetloader, text_model, tokenizer, image_model, preprocess, logit_scale, device)
    rsum_max_val = recalls_val["rSum"]
    print(f"initial rSum val:{recalls_val['rSum']}, rSum train:{recalls_train['rSum']}")
    for e in range(epochs):
        text_model.train()
        loss_sum = 0
        for image_batch, caption_batch in dataloader:
            optimizer1.zero_grad()
            with torch.set_grad_enabled(True):
                text_embs = text_model(caption_batch, tokenizer, device)
                pil_images = []
                for tensor_image in image_batch:
                    pil_image = torchvision.transforms.functional.to_pil_image(tensor_image)
                    pil_images.append(preprocess(pil_image))
                tensor_images = torch.stack(pil_images)
                with torch.no_grad():
                    image_embs = image_model(tensor_images.to(device)).float()

                logits_1 = compare_embeddings_2(image_embs, text_embs, logit_scale)
                logits_2 = compare_embeddings_2(text_embs, image_embs, logit_scale)
                label = torch.arange(len(image_embs))
                loss_1 = criterion1(logits_1, label)
                loss_2 = criterion2(logits_2, label)
                loss = (loss_1 + loss_2) / 2
                loss_sum += loss
                loss.backward()
                optimizer1.step()
        
        text_model.eval()
        recalls_train, recalls_val = \
            get_recalls(train_subsetloader, val_subsetloader, text_model, tokenizer, image_model, preprocess, logit_scale, device)
        if recalls_val['rSum'] > rsum_max_val:
            print('Best model!')
            print(recalls_val)
            rsum_max_val = recalls_val['rSum']
            torch.save(text_model.state_dict(), "mclip_finetune_text.pth")
        print(f"epoch: {e}, loss sum: {loss_sum}, rSum val: {recalls_val['rSum']}, rSum train: {recalls_train['rSum']}")


def get_recalls(train_subsetloader, val_subsetloader, text_model, tokenizer, image_model, preprocess, logit_scale, device):
    train_image_subset, train_caption_subset = next(iter(train_subsetloader))
    train_text_embs = get_text_embs(train_caption_subset, text_model, tokenizer, device)
    train_image_embs = get_image_embs(train_image_subset, image_model, preprocess, device)
    recalls_train = evaluate(train_text_embs, train_image_embs, logit_scale)
    val_image_subset, val_caption_subset = next(iter(val_subsetloader))
    val_text_embs = get_text_embs(val_caption_subset, text_model, tokenizer, device)
    val_image_embs = get_image_embs(val_image_subset, image_model, preprocess, device)
    recalls_val = evaluate(val_text_embs, val_image_embs, logit_scale)

    return recalls_train, recalls_val


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    import hdf5

    HDF5_FILE = 'coco2014.h5'
    CAPTIONS_DIR = 'captions'
    CAPTIONS_TRAIN_EN = 'captions_train2014.json'
    CAPTIONS_VAL_EN = 'captions_val2014.json'
    CAPTIONS_TRAIN_JP = 'stair_captions_v1.2_train.json'
    CAPTIONS_VAL_JP = 'stair_captions_v1.2_val.json'
    IMAGES_TRAIN_DIR = 'train2014'
    IMAGES_VAL_DIR = 'val2014'


    dataset = hdf5.HDF5dataset(HDF5_FILE)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    train_subsetloader = DataLoader(dataset, batch_size=100, shuffle=False)
    val_subsetloader = DataLoader(dataset, batch_size=100, shuffle=True)

    TEXT_MODEL_NAME = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
    IMAGE_MODEL_NAME, IMAGE_PRETRAINED_NAME = 'ViT-B-16-plus-240', "laion400m_e32"
    text_model, tokenizer, image_model, preprocess, logit_scale = \
        get_models(TEXT_MODEL_NAME, IMAGE_MODEL_NAME, IMAGE_PRETRAINED_NAME)
    finetune(dataloader, train_subsetloader, val_subsetloader, text_model, tokenizer, image_model, preprocess, logit_scale, device, epochs=20)
    print(0)
