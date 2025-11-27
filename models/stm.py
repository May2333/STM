import logging

import torch
from einops import rearrange
from torch import nn

from .backbones.vit import build_vit, build_clip
from .backbones.bert.builder import build_bert
from .criterions import MLMLoss, VTC_VTM_Loss, UTA_Loss
from .backbones.vit.actionclip import create_actionclip
from .mask import (
    TubeMaskingGenerator, 
    RandomMaskingGenerator
)
from .backbones.vit.clip import LayerNorm
from .align import AlignMask

logger = logging.getLogger(__name__)


class STM(nn.Module):
    """docstring for STM"""

    def __init__(self, config, tokenizer, is_pretrain=True):
        super(STM, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.d_model
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.itm_head = nn.Linear(self.text_width, 2)

        # criterions
        self.loss_weight = config.criterion.loss_weight
        self.criterion_uta = UTA_Loss(
            config.criterion.uta_norm_type,
            config.criterion.uta_loss_type, 
        )
        self.criterion_vtc_vtm = VTC_VTM_Loss(config.criterion.vtm_hard_neg)
        self.criterion_mlm = MLMLoss(config.criterion.mlm_masking_prob, tokenizer)

        self.uta = config.model.uta_tokens
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type

        if 'temporal' in self.video_mask_type:
            self.tau = nn.Parameter(torch.Tensor([1]))
            self.temp_attn = AlignMask(T=4, d_model=self.embed_dim, patch_size=16,input_resolution=224)
            self.temp_fc = nn.Linear(self.vision_width, self.embed_dim)
            self.temp_ln = LayerNorm(self.embed_dim)
            self.lambda_param_tp = nn.Parameter(torch.tensor(0.5))

    def forward(self, image, text, idx):
        """forward and calculate loss.  

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()

        
        
        text_embeds, pooled_text_embeds = self.encode_text(text)
        text_proj = self.text_proj(pooled_text_embeds)
        vision_embeds, pooled_vision_embeds, student_output, clip_output = self.encode_vision(image, text_proj=pooled_text_embeds)
                # obtain vision and text representations.
        vision_proj = self.vision_proj(pooled_vision_embeds)

        # calculate loss
        ## MCA loss
        if self.loss_weight.uta != 0:
            loss_uta = self.criterion_uta.uta_loss(student_output, clip_output)
        else:
            loss_uta = torch.tensor(0)

        ## VTC loss
        if self.loss_weight.vtc != 0:
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj, text_proj, idx, self.temp, all_gather=True
            )
        else:
            loss_vtc = torch.tensor(0)

        ## VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(),
                self.itm_head,
                self.temp,
                vision_embeds,
                text_embeds,
                vision_proj,
                text_proj,
                text.attention_mask,
                idx,
            )
        else:
            loss_vtm = torch.tensor(0)

        ## MLM loss
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, vision_embeds, None
            )
        else:
            loss_mlm = torch.tensor(0)

        return dict(
            loss_uta=loss_uta * self.loss_weight.uta,
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
        )
    

    def gumbel_softmax_topk(self, logits, k=1, tau=0.05, hard=False, dim=-1):
        while True:
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / tau
            y_soft = gumbels.softmax(dim)
            if (torch.isinf(gumbels).any()) or (torch.isinf(y_soft).any()) or (torch.isnan(y_soft).any()):
                print("nan!!!!!!!!!!!")
                continue
            else:
                break

        index = None
        if hard:
            _, index = torch.topk(y_soft, k)
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret, index


    def encode_teacher(self, image, uta="invisible"):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            - mask (torch.Tensor): Mask. Shape: [B,N1].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """
        B, C, T, H, W = image.shape
        mask_type = self.image_mask_type if T == 1 else self.video_mask_type
        window_size = self.image_window_size if T == 1 else self.video_window_size
        mask_ratio = self.image_mask_ratio if T == 1 else self.video_mask_ratio
        
        if self.clip_teacher is None or self.loss_weight.uta == 0:
            return None, None

        if H != self.clip_img_size:
            image = torch.nn.functional.interpolate(
                image.reshape(B, C*T, H, W), 
                size=(self.clip_img_size, self.clip_img_size), 
                mode='bicubic', align_corners=False
            )
            image = image.view(B, C, T, self.clip_img_size, self.clip_img_size)

        
        if mask_type == 'tube':
            with torch.no_grad():
                mask = TubeMaskingGenerator(window_size, mask_ratio, B)
                clip_output, attn, _ = self.clip_teacher(image)
        elif mask_type == 'random':
            with torch.no_grad():
                mask = RandomMaskingGenerator(window_size, mask_ratio, B)
                clip_output, attn, _ = self.clip_teacher(image)
        elif mask_type in 'attention':
            with torch.no_grad():
                clip_output, attn, _ = self.clip_teacher(image)
                K, _, _, _ = clip_output.shape
                BT, N = attn.shape
                N_invis = int(N * mask_ratio)
                importance = torch.multinomial(attn, N)
                mask = torch.zeros((BT, N))
                pos1 = torch.arange(BT).view(-1, 1).repeat(1, N_invis)
                pos2 = importance[:, :N_invis]
                mask[pos1, pos2] = 1
                mask = mask.view(B, -1).to(torch.bool)
        elif 'temporal' in mask_type:
            with torch.no_grad():
                clip_output, attn, vdo_feature = self.clip_teacher(image) # for teacher tokenizer
                K, _, _, _ = clip_output.shape
                clip_output = rearrange(clip_output, "k b (t n )d ->(k b n) t d", t=T)
                clip_output = self.fusion_model(clip_output)
                clip_output = rearrange(clip_output, "(k b n) t d ->k b (t n) d", b=B, t=T, k=K)
            vdo_feature = rearrange(vdo_feature, "b c t w h -> b (t w h) c ",  t=4)# for teacher tokenizer
            vdo_feature = vdo_feature.detach()
            _, B, N, D = clip_output.shape
            align_x = rearrange(vdo_feature, "b (t n) c -> b t n c",  t=4)
            align_x = self.temp_fc(align_x)
            align_x = self.temp_ln(align_x)
            former = align_x[:, :-1, :, :]
            later = align_x[:, 1:, :, :]
            _, aligned_mask = self.temp_attn(former, later)
            aligned_mask = rearrange(aligned_mask, "b t w h ->(b t) (w h)", t=T)
            aligned_mask = torch.softmax(aligned_mask, dim=-1)

            # spa
            # attn = torch.softmax(attn, dim=-1)
            aligned_mask = self.lambda_param_tp*aligned_mask + (1-self.lambda_param_tp)*attn.contiguous()
            aligned_mask = torch.softmax(aligned_mask, dim=-1)
            
            BT, N = aligned_mask.shape #[512, 196]
            N_invis =int(N * mask_ratio)
            mask, _ = self.gumbel_softmax_topk(aligned_mask, k=N_invis, hard=True)
            mask = rearrange(mask, "(b t) n ->b (t n)", t=T)
            K, _, _, C = clip_output.shape # last 6 layers
        else:
            raise NotImplementedError
            
            # mask clip output
        K, _, _, C = clip_output.shape
        mask_clip = mask.unsqueeze(0).repeat(K, 1, 1)
        # clip_output = clip_output[~mask_clip].reshape(K, B, -1, C)
        if uta == "all":
            clip_output = clip_output.reshape(K, B, -1, C)
            return mask, clip_output
        elif uta == "visible":
            clip_output = clip_output[~mask_clip].reshape(K, B, -1, C)
            return mask, clip_output
        elif uta == "visible_plus_invisibe":
            clip_output_vis = clip_output[~mask_clip].reshape(K, B, -1, C)
            clip_output_invis = clip_output[mask_clip].reshape(K, B, -1, C)
            return mask, clip_output_vis, clip_output_invis
        elif uta == "invisible":
            clip_output = clip_output.reshape(K, B, -1, C)
            return mask, clip_output
        
        return mask, clip_output

    def encode_vision(self, image, test=False,text_proj=None):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _ = self.vision_encoder(
                image, None, use_image, keep_temporal,
            )
            return vision_embeds, pooled_vision_embeds
        else:
            mask, clip_output = self.encode_teacher(image,uta=self.uta)
            if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
                keep_temporal = False
            mask_type = self.image_mask_type if T == 1 else self.video_mask_type
            vision_embeds, pooled_vision_embeds, student_output = self.vision_encoder(
                image, mask, use_image, keep_temporal, uta=self.uta, mask_type=mask_type, text_proj=text_proj
            )
            return vision_embeds, pooled_vision_embeds, student_output, clip_output

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if "vit" in encoder_name:
            vision_encoder = build_vit(self.config.model)
        else:
            raise ValueError(f"not implemented: {encoder_name}")

        teacher_name = self.config.model.vision_encoder.clip_teacher
        self.clip_teacher = None
        if teacher_name != 'none':
            self.clip_teacher, self.fusion_model = create_actionclip(self.config.model)
            # self.clip_teacher = build_clip(self.config.model)
        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_img_size
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (num_frames // tublet_size, img_size // patch_size, img_size // patch_size)
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio
        
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        
        encoder_name = self.config.model.text_encoder.name
        logger.info(f"Build text_encoder {encoder_name}")

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder
