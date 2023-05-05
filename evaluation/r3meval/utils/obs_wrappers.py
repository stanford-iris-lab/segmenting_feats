# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import gym
from gym.spaces.box import Box
import omegaconf
import torch
from torch.utils import model_zoo
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle
from torchvision.utils import save_image
import hydra


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def _get_embedding(embedding_name='resnet34', load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == 'resnet34':
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif 'resnet50' in embedding_name:
        model = models.resnet50(pretrained=True, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim

def _get_shift(shift):

    def no_shift(img):
        return img

    def bottom_left_copy_crop(img):
        img[-25:,:120,:] = img[30:55,25:145,:]
        return img

    def bottom_left_red_rectangle(img):
        img[-25:,:120,2] = 1
        return img

    def bottom_left_white_rectangle(img):
        img[-25:,:120,:] = 255
        return img

    def bottom_left_no_blue_rectangle(img):
        img[-25:,:120,2] = 1
        return img
    
    def top_right_red_rectangle(img):
        img[:40,125:,0] = 255
        return img
    
    if shift == "none":
        return no_shift
    elif shift == "bottom_left_copy_crop":
        return bottom_left_copy_crop
    elif shift == "bottom_left_red_rectangle":
        return bottom_left_red_rectangle
    elif shift == "bottom_left_white_rectangle":
        return bottom_left_white_rectangle
    elif shift == "bottom_left_no_blue_rectangle":
        return bottom_left_no_blue_rectangle
    elif shift == "top_right_red_rectangle":
        return top_right_red_rectangle
    else:
        print("Requested shift not available currently")
        raise NotImplementedError


class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class IgnoreEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m 

    def forward(self, im):
        B = im.shape[0]
        return torch.normal(torch.zeros((B, self.m)), torch.ones(B, self.m))

class MaskVisionTransformerEnc(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model

    def forward(self, im):
        B = im.shape[0]
        C = 6
        # fetch attention masks
        attn = self.vit_model.get_last_selfattention(im)
        masks = attn[:,:,0,:].reshape(B, C, -1) # B, C, H*W
        # sum weights from "good" attention masks to reweight patches
        masks = masks[:,[1,3,4]].sum(1).unsqueeze(-1)
        masks[:,0,:] = 0
        masks += 1

        # reweight tokens by masks
        x = self.vit_model.prepare_tokens(im)
        x = x * masks
        for blk in self.vit_model.blocks:
            x = blk(x)
        x = self.vit_model.norm(x)
        return x[:, 0]

class KeypointsVisionTransformerEnc(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model
        self.embed_dim = 6
    
    def get_last_value(self, im):
        x = self.vit_model.prepare_tokens(im)
        for i, blk in enumerate(self.vit_model.blocks):
            if i < len(self.vit_model.blocks) - 1:
                x = blk(x)
            else:
                # apply norm to input
                x = blk.norm1(x)
                # apply attention up to value
                B, N, C = x.shape
                qkv = blk.attn.qkv(x).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                return qkv[2]

    def forward(self, im):
        B = im.shape[0]
        C = 6
        # fetch attention masks and values
        attn = self.vit_model.get_last_selfattention(im)
        masks = attn[:,:,0,:].reshape(B, C, -1) # B, C, H*W
        values = self.get_last_value(im)
        D = values.shape[-1]
        # sum weights from "good" attention masks to reweight patches
        # masks = masks[:,[1,3,4]].sum(1).unsqueeze(-1)
        # ignore CLS token
        masks = masks[:,:,1:]
        values = values[:,:,1:]
        # find max keypoints and index into values
        keypoints = masks.argmax(-1)
        values_flat = values.reshape(B*C, -1, D) # B*C, H*W, D
        kp_flat = keypoints.reshape(-1).long()
        values_flat = values_flat[torch.arange(B*C), kp_flat]
        values = values_flat.reshape(B, C, D)
        # normalize keypoints
        keypoints = (keypoints - 98) / 196
        # return concattenated values and keypoints
        return torch.cat([values, keypoints.unsqueeze(-1)], -1).reshape(B, -1)


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env, embedding_name=None, device='cuda', load_path="", proprio=0, camera_name=None, env_name=None, shift="none"):
        gym.ObservationWrapper.__init__(self, env)

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False
        self.embedding_name = embedding_name
        if load_path == "clip":
            import clip
            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif (load_path == "random") or (load_path == "") or (embedding_name == "resnet50_insup"):
                embedding, embedding_dim = _get_embedding(embedding_name=embedding_name, load_path=load_path)
                self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif "resnet50" == embedding_name:
            from r3m import load_r3m_reproduce
            rep = load_r3m_reproduce("r3m")
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255
        elif "deit_s" in embedding_name:
            import vit_models
            if embedding_name == "deit_s_sin_dist_cls_feat":
                embedding = vit_models.dino_small_dist_cls_feat(patch_size=16, pretrained=False)
                state_dict = torch.hub.load_state_dict_from_url(url="https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth")
                msg = embedding.load_state_dict(state_dict["model"], strict=False)
                print(msg)
            elif embedding_name == "deit_s_sin_dist_shape_feat":
                embedding = vit_models.dino_small_dist_shape_feat(patch_size=16, pretrained=False)
                state_dict = torch.hub.load_state_dict_from_url(url="https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth")
                msg = embedding.load_state_dict(state_dict["model"], strict=False)
                print(msg)
            elif embedding_name == "deit_s_sin":
                embedding = vit_models.dino_small_feat(patch_size=16, pretrained=False)
                state_dict = torch.hub.load_state_dict_from_url(url="https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin.pth")
                msg = embedding.load_state_dict(state_dict["model"], strict=False)
                print(msg)
            elif embedding_name == "deit_s":
                embedding = vit_models.dino_small_feat(patch_size=16, pretrained=True)
            elif embedding_name == "deit_s_avgpool":
                embedding = vit_models.dino_small_avgpoolfeat(patch_size=16, pretrained=True)
            elif embedding_name == "deit_s_insup":
                embedding = vit_models.dino_small_feat(patch_size=16, pretrained=False)
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth")
                msg = embedding.load_state_dict(state_dict["model"], strict=False)
                print(msg)
            elif embedding_name == "deit_s_avgpool_insup":
                embedding = vit_models.dino_small_avgpoolfeat(patch_size=16, pretrained=False)
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth")
                msg = embedding.load_state_dict(state_dict["model"], strict=False)
                print(msg)


            embedding.eval()
            embedding_dim = embedding.embed_dim
            if "avgpool" in embedding_name:
                embedding_dim *= 2
            self.transforms = T.Compose([T.Resize((224, 224)),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        elif "mocov3_vits" == embedding_name:
            import vits
            embedding = vits.vit_small()
            checkpoint = model_zoo.load_url("https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar")
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            embedding.load_state_dict(state_dict, strict=False)
            embedding.head = nn.Identity()
            embedding = embedding.eval()
            embedding_dim = embedding.embed_dim
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize,
            ])
        elif "mocov3_resnet50" == embedding_name:
            embedding = models.resnet50(pretrained=False, progress=False)
            embedding_dim = 2048
            checkpoint = model_zoo.load_url("https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar")
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            embedding.load_state_dict(state_dict, strict=False)

            embedding.fc = nn.Identity()
            embedding = embedding.eval()
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize,
            ])
        elif "resnet50_sin" == embedding_name:
            embedding = models.resnet50(pretrained=False)
            embedding = embedding.eval()
            embedding_dim = 2048
            checkpoint = model_zoo.load_url('https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar')
            model =  torch.nn.DataParallel(embedding)
            # state dict is saved with DataParallel, this will change embedding weights
            model.load_state_dict(checkpoint["state_dict"]) 
            embedding.fc = Identity()
            embedding = embedding.eval()
            self.transforms = T.Compose([
                            T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor()])  

        elif "mvp" == embedding_name and "mvp" == load_path:
            import mvp
            embedding = mvp.load("vitb-mae-egosoup")
            embedding.eval()
            embedding_dim = embedding.embed_dim
            self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif "mvp" == embedding_name and "imagenet" == load_path:
            import mvp
            embedding = mvp.load("vits-sup-in")
            embedding.eval()
            embedding_dim = embedding.embed_dim
            self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif "pickle" in load_path and embedding_name == 'mvp':
            import mvp
            embedding = pickle.load(open(load_path, 'rb')).cuda()
            embedding.eval()
            embedding_dim = embedding.embed_dim
            self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif embedding_name == 'vip':
            from vip import load_vip
            embedding = load_vip().cuda()
            embedding.eval()
            embedding_dim = 1024
            self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor()]) # ToTensor() divides by 255
        elif "ignore_input" == load_path:
            self.transforms = T.Compose([T.ToTensor(),T.Resize(224)])
            embedding_dim = 1024
            embedding = IgnoreEnc(embedding_dim)
        elif "pickle" in load_path and 'dino' in embedding_name and embedding_name != 'resnet50_dino' and embedding_name != 'mask_two_pass_dino': # TODO
            # get vision transformer by loading original weights 🤪
            embedding = torch.hub.load('facebookresearch/dino:main',
                                       'dino_vits16')
            print(f"Loading model from {load_path}")
            embedding = pickle.load(open(load_path, 'rb'))
            embedding.eval()
            embedding_dim = embedding.embed_dim

            arch_args = embedding_name.split("-")
            if len(arch_args) > 1:
                state_dict = embedding.state_dict()
                new_bias = state_dict['blocks.11.attn.qkv.bias'].reshape((3, 6, -1))
                new_weight = state_dict['blocks.11.attn.qkv.weight'].reshape((3, 6, 64, 384))
                unmasked_heads = [int(um_head) for um_head in arch_args[1:]]
                for head in range(6):
                    if head not in unmasked_heads:
                        print(f"masking out {head}")
                        # surgically remove some attention maps
                        # zero out bias
                        new_bias[:,head,:] = 0
                        # zero out weight
                        new_weight[:,head,:] = 0
                state_dict['blocks.11.attn.qkv.bias'] = new_bias.reshape(-1)
                state_dict['blocks.11.attn.qkv.weight'] = new_weight.reshape((-1,384))
                embedding.load_state_dict(state_dict)

            self.transforms = T.Compose([T.ToTensor(),
                                         T.Resize(224),
                                         T.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])
        # elif "pickle" in load_path and embedding_name == 'resnet50':
        #     print(f"Loading model from {load_path}")
        #     embedding = pickle.load(open(load_path, 'rb'))
        #     embedding.eval()
        #     embedding_dim = embedding.module.outdim
        #     self.transforms = T.Compose([T.Resize(256),
        #                 T.CenterCrop(224),
        #                 T.ToTensor()]) # ToTensor() divides by 255
        elif "dino" in embedding_name and embedding_name != 'resnet50_dino' and embedding_name != 'mask_two_pass_dino':
            embedding = torch.hub.load('facebookresearch/dino:main',
                                       'dino_vits16')
            embedding.eval()
            embedding_dim = embedding.embed_dim
            if embedding_name == "dino_ensemble":
                num_heads = 6
                ensemble_weights = torch.FloatTensor(num_heads) # WARNING: Variable not added to module's parameter list
                ensemble_weights[:] = embedding.blocks[-1].attn.scale # TODO: add noise, should not suffer from symmetry tho
                ensemble_weights = ensemble_weights.reshape((1, -1, 1, 1)).cuda() # reshape to match attn to avoid broadcasting in Attention
                embedding.blocks[-1].attn.scale = ensemble_weights
                ensemble_weights.requires_grad = True

            arch_args = embedding_name.split("-")
            if len(arch_args) > 1:
                state_dict = embedding.state_dict()
                new_bias = state_dict['blocks.11.attn.qkv.bias'].reshape((3, 6, -1))
                new_weight = state_dict['blocks.11.attn.qkv.weight'].reshape((3, 6, 64, 384))
                unmasked_heads = [int(um_head) for um_head in arch_args[1:]]
                for head in range(6):
                    if head not in unmasked_heads:
                        print(f"masking out {head}")
                        # surgically remove some attention maps
                        # zero out bias
                        new_bias[:,head,:] = 0
                        # zero out weight
                        new_weight[:,head,:] = 0
                state_dict['blocks.11.attn.qkv.bias'] = new_bias.reshape(-1)
                state_dict['blocks.11.attn.qkv.weight'] = new_weight.reshape((-1,384))
                embedding.load_state_dict(state_dict)

            self.transforms = T.Compose([T.ToTensor(),
                                         T.Resize(224),
                                         T.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])
        elif embedding_name=='resnet50_dino':
            embedding = torch.hub.load('facebookresearch/dino:main',
                                       'dino_resnet50')
            
            embedding.eval()
            embedding_dim = 2048

            self.transforms = T.Compose([T.Resize(256, interpolation=3),
                                         T.CenterCrop(224),
                                         T.ToTensor(),
                                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        # elif embedding_name=='resnet50_dino' and 'pickle' in load_path:
        #     try:
        #         print(f"Loading model from {load_path}, resnet50_dino")
        #         embedding = pickle.load(open(load_path, 'rb'))
        #     except:
        #         # /iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/2022-11-01_16-28-13/
        #         import pdb; pdb.set_trace()
            
        #     embedding.eval()
        #     embedding_dim = embedding.embed_dim

        #     self.transforms = T.Compose([T.Resize(256, interpolation=3),
        #                                  T.CenterCrop(224),
        #                                  T.ToTensor(),
        #                                  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        elif embedding_name == 'mask_two_pass_dino':
            vit_model = torch.hub.load('facebookresearch/dino:main',
                                       'dino_vits16')
            vit_model.eval()
            embedding_dim = vit_model.embed_dim
            embedding = MaskVisionTransformerEnc(vit_model)
            embedding.eval()

            self.transforms = T.Compose([T.ToTensor(),
                                         T.Resize(224),
                                         T.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])
        elif embedding_name == 'keypoints':
            import dino
            vit_model = torch.hub.load('facebookresearch/dino:main',
                                       'dino_vits16')
            vit_model.eval()
            embedding_dim = 6*65
            embedding = KeypointsVisionTransformerEnc(vit_model)
            embedding.eval()

            self.transforms = T.Compose([T.ToTensor(),
                                         T.Resize(224),
                                         T.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])
        else:
            raise NameError("Invalid Model")
        embedding.eval()

        if device == 'cuda' and torch.cuda.is_available():
            print('Using CUDA.')
            device = torch.device('cuda')
        else:
            print('Not using CUDA.')
            device = torch.device('cpu')
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=(self.embedding_dim+self.proprio,))

    def observation(self, observation):
        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None:
            inp = self.transforms(Image.fromarray(observation.astype(np.uint8))).reshape(-1, 3, 224, 224)
            if not ('VisionTransformer' in type(self.embedding).__name__ or 'moco' in self.embedding_name): # "r3m" in self.load_path and "pickle" not in self.load_path:
                print("shifting input to 0-255 (should only happen for R3M)")
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                inp *= 255.0
            inp = inp.to(self.device)
            with torch.no_grad():
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()

            ## IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[:self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[:self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def encode_batch(self, obs, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp = []
        for o in obs:
            i = self.transforms(Image.fromarray(o.astype(np.uint8))).reshape(-1, 3, 224, 224)
            if (self.embedding_name == 'resnet50') or (self.embedding_name == 'resnet50_insup') or (self.embedding_name == 'resnet50_dino'): # mapping resnet50 to R3M # if not 'VisionTransformer' in type(self.embedding).__name__: # and "pickle" not in self.load_path: # not 'VisionTransformer' in type(self.embedding).__name__: 
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                print("shifting input to 0-255 (should only happen for resnets)")
                i *= 255.0
            inp.append(i)
        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb =  self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None:
            return self.observation(self.env.observation(None))
        else:
            # returns the state based observations
            return self.env.unwrapped.get_obs()
          
    def start_finetuning(self):
        self.start_finetune = True


class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, shift="none", *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        self.shift = _get_shift(shift)
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

    def get_image(self):
        if self.camera_name == "default":
            print("Camera not supported")
            assert(False)
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                            device_id=self.device_id)
        else:
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                              camera_name=self.camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        img = self.shift(img)

        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()
        