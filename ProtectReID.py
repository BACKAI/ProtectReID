
import os
import math
import faiss
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.log_utils import save_w
from criteria import l2_loss
from reid_model.model import ft_net
from training.coaches.base_coach import BaseCoach
from configs import paths_config, hyperparameters, global_config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode, Normalize


class ProtectReID(BaseCoach):
    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)


    def MI_FGSM_vis(
        self,
        generator,
        lc_coarse,
        lc_fine,
        bot_image,
        real_images_batch,
        alpha,
        momentum,
        num_iter
    ):
        adv_latent = lc_coarse.clone().detach()
        grad_m = torch.zeros_like(adv_latent)

        for i in range(num_iter):
            adv_latent.requires_grad_(True)
            latent = torch.cat([adv_latent, lc_fine], dim=1)
            img_adv  = generator.synthesis(latent, noise_mode='const')
            d_pos = l2_loss.l2_loss(img_adv, bot_image)
            d_neg = l2_loss.l2_loss(img_adv, real_images_batch)
            margin = 0.5
            margin_loss = F.relu(d_pos - d_neg + margin)
            margin_loss.backward()
            g = adv_latent.grad.data
            norm = g.abs().view(g.shape[0], -1).mean(dim=1, keepdim=True)
            g = g / norm.clamp(min=1e-12).view(-1,1,1)
            grad_m = momentum * grad_m + g
            adv_latent.data += alpha/8 * grad_m.sign()
            adv_latent.data = adv_latent.data
            adv_latent.grad.zero_()
        return adv_latent.detach()


    def MI_FGSM_ID(
        self,
        generator,
        reid_model,
        lc_modified_coarse,
        lc_fine,
        feat_orig,
        alpha,
        momentum,
        num_iter
    ):
        adv_latent = lc_fine.clone().detach()
        grad_m = torch.zeros_like(adv_latent)

        for i in range(num_iter):
            adv_latent.requires_grad_(True)
            latent = torch.cat([lc_modified_coarse, adv_latent], dim=1)
            img_adv  = generator.synthesis(latent, noise_mode='const')
            gimg = (resize(img_adv, size=(256, 128), interpolation=InterpolationMode.BICUBIC) + 1) / 2
            imagenet_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            gimg = imagenet_normalize(gimg)
            feat_adv = reid_model(gimg)
            cos_sim = F.cosine_similarity(feat_adv, feat_orig, dim=1)
            loss = 1.0 - cos_sim.mean()
            loss.backward()
            g = adv_latent.grad.data
            norm = g.abs().view(g.shape[0], -1).mean(dim=1, keepdim=True)
            g = g / norm.clamp(min=1e-12).view(-1,1,1)
            grad_m = momentum * grad_m + g
            adv_latent.data -= alpha * grad_m.sign()
            adv_latent.data = adv_latent.data
            adv_latent.grad.zero_()
        return adv_latent.detach()
    
#----------------------------------------------------

    def train(self):
        reid_transform = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        reidmodel = ft_net(751).to(global_config.device)
        save_path = os.path.join("../pretrained_model/resnet50_reid.pth")
        reidmodel.load_state_dict(torch.load(save_path), strict=False)
        reidmodel.classifier.classifier = nn.Sequential()
        reidmodel = reidmodel.eval()
        for p in reidmodel.parameters():
            p.requires_grad = False

        result = scipy.io.loadmat("../pretrained_model/Vector_gallery.mat")
        gallery_e4e = torch.FloatTensor(result['gallery_e4e']).to(global_config.device)
        gallery_reid = torch.FloatTensor(result['gallery_f']).cpu().numpy()
        gallery_reid = np.ascontiguousarray(gallery_reid, dtype=np.float32)
        faiss.normalize_L2(gallery_reid)

#----------------------------------------------------

        # Identity to latent code retrieval
        for _, (fname, image, img_path) in tqdm(enumerate(self.data_loader)):
            img = Image.open(img_path[0]).convert('RGB')
            img = reid_transform(img)
            img = img.unsqueeze(0).to(global_config.device)
            with torch.no_grad():
                outputs = reidmodel(img)
            outputs = outputs.detach().cpu().numpy()
            outputs = np.ascontiguousarray(outputs, dtype=np.float32)
            faiss.normalize_L2(outputs)
            index = faiss.IndexFlatIP(512)
            index.add(gallery_reid)
            top_k = 10
            _, top_indices = index.search(outputs, top_k)
            _, bot_indices = index.search(-outputs, top_k)
            ti = torch.as_tensor(top_indices, device=gallery_e4e.device, dtype=torch.long)
            bi = torch.as_tensor(bot_indices, device=gallery_e4e.device, dtype=torch.long)

            selected_latent_t = gallery_e4e[ti].mean(dim=1)
            selected_latent_b = gallery_e4e[bi].mean(dim=1)
            
#----------------------------------------------------

            # Reciprocal self-attention module
            n, c, d = selected_latent_t.shape
            Q = K = V = selected_latent_t.permute(1, 0, 2)
        
            trans_weights = torch.matmul(Q, K.transpose(2, 1)) / math.sqrt(d)
            inv_weights = 1.0 / trans_weights
            tmp = 0.1
            soft_weights = F.softmax(inv_weights/tmp, dim=-1)
            sum_weights = torch.matmul(soft_weights, V)
            weights = sum_weights.permute(1, 0, 2)
            least_similar = torch.mean(weights, dim=0, keepdim=True)
            channel_indices = torch.randperm(14, device=least_similar.device)
            top_lc = least_similar[:, channel_indices, :]
            bot_lc = torch.mean(selected_latent_b, dim=0, keepdim=False)
            bot_lc = bot_lc.unsqueeze(0)

            image_name = fname[0]

            real_images_batch = image.to(global_config.device)
            with torch.no_grad():
                rimg = (resize(real_images_batch, size=(256, 128),
                               interpolation=InterpolationMode.BICUBIC) + 1) / 2
                imagenet_normalize = Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                rimg = imagenet_normalize(rimg)
                orig_feat = reidmodel(rimg)

#----------------------------------------------------

            # Hierarchical latent code manipulation
            with torch.no_grad():
                bot_image = self.G.synthesis(bot_lc, noise_mode='const').detach()
            os.makedirs("../output", exist_ok=True)

            sep = 4
            lc_coarse = top_lc[:, :sep, :].clone()
            lc_fine = top_lc[:, sep:, :].clone() 

            lc_modified_coarse = self.MI_FGSM_vis(
                self.G,
                lc_coarse,
                lc_fine,
                bot_image,
                real_images_batch,
                alpha=hyperparameters.alpha,
                momentum=hyperparameters.momentum,
                num_iter=hyperparameters.num_iter
            )

            lc_modified_fine = self.MI_FGSM_ID(
                self.G,
                reidmodel,
                lc_modified_coarse,
                lc_fine,
                orig_feat,
                alpha=hyperparameters.alpha,
                momentum=hyperparameters.momentum,
                num_iter=hyperparameters.num_iter
            )

            final1 = torch.cat([lc_modified_coarse, lc_modified_fine], dim=1)
            save_w(final1, self.G, image_name, '', "../output", resize_to=(128,64))