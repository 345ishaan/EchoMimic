#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EchoMimic
@File    ：audio2vid.py
@Author  ：juzhen.czy
@Date    ：2024/3/4 17:43 
'''
import argparse
import os
import requests
import random
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import uuid
import json
from typing import List, Dict, Optional, Any
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import is_accelerate_available
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from omegaconf import OmegaConf
from PIL import Image

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN

assert is_accelerate_available(), "accelarate not available"

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/animation.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--facemusk_dilation_ratio", type=float, default=0.1)
    parser.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)

    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="./cache")

    args = parser.parse_args()

    return args

def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]

def get_bbox_from_cache(image_url: str, cache_dir: str, fname="bbox.json"):
    if not os.path.exists(os.path.join(cache_dir, fname)):
        return None
    fp = open(os.path.join(cache_dir, fname))
    try:
        data = json.loads(fp.read())
    except Exception as e:
        print(e)
        return None
    if image_url in data:
        return data[image_url]
    return None

def write_bbox_to_cache(image_url: str, cache_dir: str, bbox: List[Any], fname="bbox.json"):
    fname = os.path.join(cache_dir, fname)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data = {}
    data[image_url] = bbox
    if not os.path.exists(fname):
        fp = open(fname, 'w')
        json.dump(data, fp)
    else:
        cur_data = open(fname).read()
        if cur_data:
            data.update(json.loads(cur_data))
        json.dump(data, open(fname, 'w'))


class GenimeLipSync:
    
    def __init__(self):
        args = parse_args()
        self.args = args

        config = OmegaConf.load(args.config)
        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
        self.weight_dtype = weight_dtype

        device = args.device
        if device.__contains__("cuda") and not torch.cuda.is_available():
            device = "cpu"

        inference_config_path = config.inference_config
        infer_config = OmegaConf.load(inference_config_path)


        ############# model_init started #############

        ## vae init
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to("cuda", dtype=weight_dtype)

        ## reference net init
        reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device=device)
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )

        ## denoising net init
        if os.path.exists(config.motion_module_path):
            ### stage1 + stage2
            denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                config.pretrained_base_model_path,
                config.motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=infer_config.unet_additional_kwargs,
            ).to(dtype=weight_dtype, device=device)
        else:
            ### only stage1
            denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                config.pretrained_base_model_path,
                "",
                subfolder="unet",
                unet_additional_kwargs={
                    "use_motion_module": False,
                    "unet_use_temporal_attention": False,
                    "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
                }
            ).to(dtype=weight_dtype, device=device)
        denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False
        )

        ## face locator init
        face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device="cuda"
        )
        face_locator.load_state_dict(torch.load(config.face_locator_path))

        ### load audio processor params
        audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

        ### load face detector params
        face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

        ############# model_init finished #############

        width, height = args.W, args.H
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        self.pipe = Audio2VideoPipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            audio_guider=audio_processor,
            face_locator=face_locator,
            scheduler=scheduler,
        )
        self.pipe = self.pipe.to("cuda", dtype=weight_dtype)
        self.pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)


    def infer(self, image_urls: List[str], audio_urls: List[str], save_dir: str):
        save_video_fnames = []
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx, (img_url, audio_url) in enumerate(zip(image_urls, audio_urls)):
            response = requests.get(img_url)
            if response.status_code == 200:
                image_data = np.asarray(bytearray(response.content), dtype="uint8")
                face_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                import pdb
                #pdb.set_trace()
                assert face_img is not None
            else:
                raise Exception(f"Could not decoded image blob url {img_url}")
            audio_res = requests.get(audio_url)
            if audio_res.status_code == 200:
                audio_path = os.path.join(save_dir, f"audio_{idx}.wav")
                fp = open(audio_path, "wb")
                fp.write(audio_res.content)
                fp.close()
            else:
                raise Exception("Unable to decode audio blob url")
    
            if self.args.seed is not None and self.args.seed > -1:
                generator = torch.manual_seed(self.args.seed)
            else:
                generator = torch.manual_seed(random.randint(100, 1000000))
    
            final_fps = self.args.fps
    
            #### face mask prepare
    
            # check if the url is present in the cache.
    
            face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    
            select_bbox = get_bbox_from_cache(img_url, self.args.cache_dir)
            if select_bbox is None:
                det_bboxes, probs = face_detector.detect(face_img)
                if det_bboxes is None or probs is None:
                    det_bboxes, probs = face_detector.detect(face_img[:,:,::-1])
                select_bbox = select_face(det_bboxes, probs)
    
            if select_bbox is None:
                face_mask[:, :] = 255
            else:
                xyxy = select_bbox[:4]
                xyxy = np.round(xyxy).astype('int')
                # write to cache.
                write_bbox_to_cache(img_url, self.args.cache_dir, xyxy.tolist())
                rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
                r_pad = int((re - rb) * self.args.facemusk_dilation_ratio)
                c_pad = int((ce - cb) * self.args.facemusk_dilation_ratio)
                face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255
    
                #### face crop
                r_pad_crop = int((re - rb) * self.args.facecrop_dilation_ratio)
                c_pad_crop = int((ce - cb) * self.args.facecrop_dilation_ratio)
                crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + c_pad_crop, face_img.shape[0])]
                print(crop_rect)
                face_img = crop_and_pad(face_img, crop_rect)
                face_mask = crop_and_pad(face_mask, crop_rect)
                face_img = cv2.resize(face_img, (self.args.W, self.args.H))
                face_mask = cv2.resize(face_mask, (self.args.W, self.args.H))
    
            ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
            face_mask_tensor = torch.Tensor(face_mask).to(dtype=self.weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
            
            stime = time.time()
    
            video = self.pipe(
                ref_image_pil,
                audio_path,
                face_mask_tensor,
                self.args.W,
                self.args.H,
                self.args.L,#video length
                10,#args.steps, # inference steps
                2.5,#args.cfg, # guidance scale
                generator=generator,
                audio_sample_rate=self.args.sample_rate,
                context_frames=self.args.context_frames,
                fps=final_fps,
                context_overlap=self.args.context_overlap
            ).videos
    
            print("Done running pipeline inference in {}".format(time.time() - stime))
    
            stime = time.time()
            video_fname = os.path.join(save_dir, f"video_{idx}.mp4")
            video_w_audio_fname = os.path.join(save_dir, f"video_audio_{idx}.mp4")
            video = video
            save_videos_grid(
                video,
                video_fname,
                n_rows=1,
                fps=final_fps,
            )
    
            video_clip = VideoFileClip(video_fname)
            audio_clip = AudioFileClip(audio_path)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(video_w_audio_fname,
                    codec="libx264", audio_codec="aac")
            print("Done Video Processing took: {}".format(time.time()-stime))
            save_video_fnames.append(video_w_audio_fname)
        concat_file = os.path.join(save_dir, 'concat_list.txt')
        final_concat_fname = os.path.join(save_dir, 'final_concat.mp4')
        with open(concat_file, 'w') as f:
            for fname in save_video_fnames:
                f.write(f"file '{fname}'\n")
        concat_cmd = f"/home/EchoMimic/ffmpeg-7.0.1-amd64-static/ffmpeg -y -safe 0 -f concat -i {concat_file} -c copy {final_concat_fname}"
        os.system(concat_cmd)

if __name__ == "__main__":
    lipsync = GenimeLipSync()

    img_urls = [
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_sam.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9zYW0ud2VicCIsImlhdCI6MTcyMTQ3MDgzNSwiZXhwIjoxNzUzMDA2ODM1fQ.GXTUB7iYGkrEQIDJahtkdLFyInyetHgfSv5hgHPtvSk&t=2024-07-20T10%3A20%3A35.248Z",
        #"https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_sam.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9zYW0ud2VicCIsImlhdCI6MTcyMTQ3MDgzNSwiZXhwIjoxNzUzMDA2ODM1fQ.GXTUB7iYGkrEQIDJahtkdLFyInyetHgfSv5hgHPtvSk&t=2024-07-20T10%3A20%3A35.248Z"
    ]
    audio_urls = [
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/170.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzE3MC53YXYiLCJpYXQiOjE3MjE0OTk3OTQsImV4cCI6MTc1MzAzNTc5NH0.Eb6UZMRhlQc_dn308U-2Qnqq-PJAcxfFq-qqE4lVsWg&t=2024-07-20T18%3A23%3A14.638Z",
        #"https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/169.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzE2OS53YXYiLCJpYXQiOjE3MjE0OTk5NzEsImV4cCI6MTc1MzAzNTk3MX0.PbIuOPDorH-0z7yvAMjk_kexeFeHNPxKo-D8JG-rfr4&t=2024-07-20T18%3A26%3A11.485Z"
    ]
    save_dir = "/home/EchoMimic/genime_results"
    lipsync.infer(img_urls, audio_urls, save_dir)
