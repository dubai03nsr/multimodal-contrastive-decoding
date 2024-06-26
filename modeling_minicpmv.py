import math
from typing import List, Optional
import json

import timm
import torch
import torch.nn.functional as F
import torchvision
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
from transformers import LlamaTokenizer

from .configuration_minicpm import MiniCPMVConfig
from .modeling_minicpm import MiniCPMPreTrainedModel, MiniCPMForCausalLM
from .resampler import Resampler


class MiniCPMVPreTrainedModel(MiniCPMPreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm = MiniCPMForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim ,self.vision_dim)
        self.transform = self.init_transform()

        self.plaus_hp = 0.1
        # self.txt_hp = 0.5
        # self.img_hp = 0.5
        # print('plaus_hp', self.plaus_hp, 'txt_hp', self.txt_hp, 'img_hp', self.img_hp)

    def init_vision_module(self):
        model = timm.create_model(
            self.config.vision_encoder,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True
        )

        if isinstance(model, timm.models.VisionTransformer):
            if model.attn_pool is not None:
                model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            grid_size=int(math.sqrt(self.config.query_num)),
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
        )

    def init_transform(self):
        return transforms.Compose([
            transforms.Resize(
                (self.config.image_size, self.config.image_size),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])



    def get_vision_embedding(self, pixel_values):
        res = []
        dtype = self.vpm.pos_embed.data.dtype
        for pixel_value in pixel_values:
            vision_embedding = self.vpm.forward_features(pixel_value.unsqueeze(0).type(dtype))
            if hasattr(self.vpm, 'num_prefix_tokens') and self.vpm.num_prefix_tokens > 0:
                vision_embedding = vision_embedding[:, self.vpm.num_prefix_tokens:]
            res.append(self.resampler(vision_embedding))
        return torch.vstack(res)

    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values))
                elif self.training:
                    dtype = self.vpm.pos_embed.data.dtype
                    device = self.vpm.pos_embed.data.device
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    vision_hidden_states.append(self.get_vision_embedding(dummy_image))
                else:
                    vision_hidden_states.append([])

        else:
            vision_hidden_states = data['vision_hidden_states']

        vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )


    def _convert_to_tensors(self, tokenizer, input_str, max_inp_length: Optional[int] = None):
        if tokenizer.add_bos_token:
            input_ids = tokenizer.encode(input_str)
        else:
            input_ids = [tokenizer.bos_id] + tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[: max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[0]
        # 跳过 im_start
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bound = torch.hstack(
            [image_start_tokens[: valid_image_nums].unsqueeze(-1),
             image_end_tokens[:valid_image_nums].unsqueeze(-1)]
        )

        model_input = {}
        model_input["input_ids"] = input_ids.unsqueeze(0).to(self.device)
        model_input["image_bound"] = image_bound

        return model_input


    def _process_list(self, tokenizer, data_list: List[str], max_inp_length: Optional[int] = None):
        pad_keys = ['input_ids']
        input_tensors = []
        for data in data_list:
            input_tensors.append(self._convert_to_tensors(tokenizer, data, max_inp_length))
        padded = {}
        for key in pad_keys:
            padded[key] = pad(input_tensors, key, padding_side="left").to(self.device)
        padded['image_bound'] = [i['image_bound'] for i in input_tensors]
        return padded

    def _decode(self, inputs_embeds, tokenizer, **kwargs):
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        return self._decode_text(output, tokenizer)

    def _Decode(self, inputs_embeds, tokenizer, **kwargs):
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
        )
        result_ids = self._Decode_text(output.sequences, tokenizer)
        return result_ids, output.scores

    def _decode_text(self, result_ids, tokenizer):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] == tokenizer.eos_id:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def _Decode_text(self, result_ids, tokenizer):
        result_ids_ = []
        for result in result_ids:
            result = result[result != 0]
            if result.shape[0] > 0 and result[0] == tokenizer.bos_id:
                result = result[1:]
            if result.shape[0] > 0 and result[-1] == tokenizer.eos_id:
                result = result[:-1]
            result_ids_.append(result)
        return result_ids_

    def generate(
            self,
            data_list=None,
            img_list=None,
            tokenizer=None,
            max_inp_length: Optional[int] = None,
            vision_hidden_states=None,
            return_vision_hidden_states=False,
            **kwargs
    ):

        assert data_list is not None
        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self._process_list(tokenizer, data_list, max_inp_length)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(self.transform(img))
                if img_inps:
                    pixel_values.append(torch.stack(img_inps).to(self.device))
                else:
                    pixel_values.append([])
            model_inputs['pixel_values'] = pixel_values
        else:
            model_inputs['vision_hidden_states'] = vision_hidden_states

        with torch.inference_mode():
            model_inputs['inputs_embeds'], vision_hidden_states = self.get_vllm_embedding(model_inputs)

            result = self._decode(model_inputs['inputs_embeds'], tokenizer, **kwargs)

        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result

    def Generate(
            self,
            data_list=None,
            img_list=None,
            tokenizer=None,
            max_inp_length: Optional[int] = None,
            vision_hidden_states=None,
            return_vision_hidden_states=False,
            **kwargs
    ):

        assert data_list is not None
        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self._process_list(tokenizer, data_list, max_inp_length)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(self.transform(img))
                if img_inps:
                    pixel_values.append(torch.stack(img_inps).to(self.device))
                else:
                    pixel_values.append([])
            model_inputs['pixel_values'] = pixel_values
        else:
            model_inputs['vision_hidden_states'] = vision_hidden_states

        with torch.inference_mode():
            model_inputs['inputs_embeds'], vision_hidden_states = self.get_vllm_embedding(model_inputs)

            result, scores = self._Decode(model_inputs['inputs_embeds'], tokenizer, **kwargs)

        if return_vision_hidden_states:
            return result, scores, vision_hidden_states

        return result, scores

    def chat(self, image, msgs, context, tokenizer, vision_hidden_states=None, max_new_tokens=2048, sampling=False, **kwargs):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # msgs to prompt
        prompt = ''
        for i, msg in enumerate(msgs):
            role = msg['role']
            content = msg['content']
            assert role in ['user', 'assistant']
            if i == 0:
                assert role == 'user', 'The role of first msg should be user'
                content = tokenizer.im_start + tokenizer.unk_token * self.config.query_num + tokenizer.im_end + '\n' + content
            prompt += '<用户>' if role=='user' else '<AI>'
            prompt += content
        prompt += '<AI>'
        final_input = prompt

        if sampling:
            generation_config = {
                'top_p': 0.8,
                'top_k': 100,
                'temperature':0.6,
                'do_sample': True
            }
        else:
            generation_config = {
                'num_beams': 3,
                'repetition_penalty': 1.2,
            }

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        with torch.inference_mode():
            res, vision_hidden_states = self.generate(
                data_list=[final_input],
                max_inp_length=2048,
                img_list=[[image]],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                **generation_config
            )
        answer = res[0]
        context = msgs
        context.append({'role':'assistant', 'content': answer})

        return answer, context, generation_config

    def Chat(self, image, src_text, tokenizer, tgt_lang='en', txt_hp=0.0, img_hp=0.0, vision_hidden_states=None, **kwargs):
        print('txt_hp', txt_hp, 'img_hp', img_hp)

        pre_prompt = tokenizer.im_start + tokenizer.unk_token * self.config.query_num + tokenizer.im_end + '\n<用户>'
        post_prompt = '\n<AI>'
        if tgt_lang == 'en':
            prompts = {
                # 'exp': pre_prompt + f'Here is the Chinese caption of the image: {src_text}\nDescribe the image in English.' + post_prompt,
                'exp': pre_prompt + f'Here is the Chinese caption of the image: {src_text}\nDescribe the image in 1 sentence in English.' + post_prompt,
                'txt': pre_prompt + f'Translate this to English: {src_text}' + post_prompt,
                # 'txt': pre_prompt + f'Translate this to English in 1 sentence: {src_text}' + post_prompt,
                # 'img': pre_prompt + f'Describe the image.' + post_prompt
                'img': pre_prompt + f'Describe the image in 1 sentence.' + post_prompt
            }
        elif tgt_lang == 'zh':
            prompts = {
                # 'exp': pre_prompt + f'这是图像的英文说明：{src_text}\n用中文描述这幅图像。' + post_prompt,
                'exp': pre_prompt + f'这是图像的英文说明：{src_text}\n用1句话中文描述这幅图像。' + post_prompt,
                'txt': pre_prompt + f'翻译成中文：{src_text}' + post_prompt,
                # 'img': pre_prompt + f'描述这幅图像。' + post_prompt
                'img': pre_prompt + f'用1句话描述这幅图像。' + post_prompt
            }
        else: assert(False)

        with torch.inference_mode():
            gen_ids = []

            for _ in range(1000):
                # exp
                res_exp, scores_exp, vision_hidden_states = self.Generate(
                    data_list=[prompts['exp'] + tokenizer.decode(gen_ids)],
                    max_inp_length=2048,
                    img_list=[[image]],
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    vision_hidden_states=vision_hidden_states,
                    return_vision_hidden_states=True
                )
                if len(res_exp[0]) == 0:
                    break
                probs_exp = torch.softmax(scores_exp[0][0], dim=0)
                max_prob = probs_exp.max()
                logprobs_exp = F.log_softmax(scores_exp[0][0], dim=0)
                logprobs_exp[probs_exp < max_prob * self.plaus_hp] = float('-inf')

                # """
                # txt
                res_txt, scores_txt = self.Generate(
                    data_list=[prompts['txt'] + tokenizer.decode(gen_ids)],
                    max_inp_length=2048,
                    img_list=None,
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    vision_hidden_states=None,
                    return_vision_hidden_states=False
                )
                if len(res_txt[0]) == 0:
                    logprobs_txt = torch.zeros_like(logprobs_exp)
                else:
                    logprobs_txt = F.log_softmax(scores_txt[0][0], dim=0)
                # """

                """
                # img
                res_img, scores_img = self.Generate(
                    data_list=[prompts['img'] + tokenizer.decode(gen_ids)],
                    max_inp_length=2048,
                    img_list=[[image]],
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    vision_hidden_states=vision_hidden_states,
                    return_vision_hidden_states=False
                )
                if len(res_img[0]) == 0:
                    logprobs_img = torch.zeros_like(logprobs_exp)
                else:
                    logprobs_img = F.log_softmax(scores_img[0][0], dim=0)
                """

                # combine
                # logprobs = logprobs_exp - txt_hp * logprobs_txt - img_hp * logprobs_img
                logprobs = logprobs_exp - txt_hp * logprobs_txt
                # logprobs = logprobs_exp - img_hp * logprobs_img
                argmax_id = torch.argmax(logprobs)
                gen_ids.append(argmax_id)

            return tokenizer.decode(gen_ids), vision_hidden_states

        """
        answer = res[0]
        context = msgs
        context.append({'role':'assistant', 'content': answer})

        return answer, context, scores
        """

class LlamaTokenizerWrapper(LlamaTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"

    @property
    def eos_id(self):
        return self.sp_model.eos_id()

    @property
    def bos_id(self):
        return self.sp_model.bos_id()

    @property
    def unk_id(self):
        return self.sp_model.unk_id()

    @property
    def im_start_id(self):
        return self._convert_token_to_id(self.im_start)

    @property
    def im_end_id(self):
        return self._convert_token_to_id(self.im_end)


def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor

