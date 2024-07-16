#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
webui
'''

import os
import random
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
from facenet_pytorch import MTCNN
import argparse

import gradio as gr

from src.utils.mp_utils import LMKExtractor
from src.utils.draw_utils import FaceMeshVisualizer
from src.utils.img_utils import pil_to_cv2, center_crop_cv2, pils_from_video, save_video_from_cv2_list
from src.utils.motion_utils import motion_sync

default_values = {
    "width": 512,
    "height": 512,
    "length": 1200,
    "seed": 420,
    "facemask_dilation_ratio": 0.1,
    "facecrop_dilation_ratio": 0.5,
    "context_frames": 12,
    "context_overlap": 3,
    "cfg": 2.5,
    "steps": 30,
    "sample_rate": 16000,
    "fps": 24,
    "device": "cuda"
}

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

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

def process_video(uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio, facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device):
    config_path = "./configs/prompts/animation.yaml"
    config = OmegaConf.load(config_path)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = "cuda"
    if not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    ############# model_init started #############
    ## vae init
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to("cuda", dtype=weight_dtype)

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))

    ## denoising_unet init
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

    denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)

    ## face locator init
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
    face_locator.load_state_dict(torch.load(config.face_locator_path))

    ## load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    ## load face detector params
    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

    ############# model_init finished #############

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    ).to("cuda", dtype=weight_dtype)
    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))

    #### face musk prepare
    face_img = cv2.imread(uploaded_img)
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4]
        xyxy = np.round(xyxy).astype('int')
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * facemask_dilation_ratio)
        c_pad = int((ce - cb) * facemask_dilation_ratio)
        face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255
        
        #### face crop
        r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
        face_img = crop_and_pad(face_img, crop_rect)
        face_mask = crop_and_pad(face_mask, crop_rect)
        face_img = cv2.resize(face_img, (width, height))
        face_mask = cv2.resize(face_mask, (width, height))

    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
    
    video = pipe(
        ref_image_pil,
        uploaded_audio,
        face_mask_tensor,
        width,
        height,
        length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap
    ).videos

    save_dir = Path("output/tmp")
    save_dir.mkdir(exist_ok=True, parents=True)
    output_video_path = save_dir / "output_video.mp4"
    save_videos_grid(video, str(output_video_path), n_rows=1, fps=fps)

    video_clip = VideoFileClip(str(output_video_path))
    audio_clip = AudioFileClip(uploaded_audio)
    final_output_path = save_dir / "output_video_with_audio.mp4"
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(str(final_output_path), codec="libx264", audio_codec="aac")

    return str(final_output_path)

def process_motion_sync(video_path, ref_image_path, uploaded_audio, width, height, fps):
    vis = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True, draw_pupil=True)
    lmk_extractor = LMKExtractor()

    input_frames_cv2 = [cv2.resize(center_crop_cv2(pil_to_cv2(i)), (width, height)) for i in pils_from_video(video_path)]
    ref_frame = cv2.resize(cv2.imread(ref_image_path), (width, height))
    ref_det = lmk_extractor(ref_frame)

    sequence_driver_det = []
    for frame in input_frames_cv2:
        result = lmk_extractor(frame)
        if result is not None:
            sequence_driver_det.append(result)
        else:
            print("face detection failed")
            return "Face detection failed for some frames"

    pose_frames_driver = [vis.draw_landmarks((width, height), i["lmks"], normed=True) for i in sequence_driver_det]
    poses_add_driver = [(i * 0.5 + j * 0.5).clip(0, 255).astype(np.uint8) for i, j in zip(input_frames_cv2, pose_frames_driver)]

    save_dir = './output/{}'.format(os.path.basename(ref_image_path).replace('.png', ''))
    os.makedirs(save_dir, exist_ok=True)

    sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
    for i in range(len(sequence_det_ms)):
        with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
            pickle.dump(sequence_det_ms[i], file)

    pose_frames = [vis.draw_landmarks((width, height), i, normed=False) for i in sequence_det_ms]  # 移除 white 参数
    poses_add = [(i * 0.5).clip(0, 255).astype(np.uint8) for i in pose_frames]  # 去掉原始图像背景

    poses_cat = [np.concatenate([i, j], axis=1) for i, j in zip(poses_add_driver, poses_add)]

    final_pose_video_path = save_dir + "/vis_example_with_pose.mp4"
    save_video_from_cv2_list(poses_cat, final_pose_video_path, fps=fps)

    # 添加音频
    video_clip = VideoFileClip(final_pose_video_path)
    audio_clip = AudioFileClip(uploaded_audio)
    final_output_with_audio_path = save_dir + "/vis_example_with_pose_audio.mp4"
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(final_output_with_audio_path, codec="libx264", audio_codec="aac")

    return final_output_with_audio_path

with gr.Blocks() as demo:
    gr.Markdown('# EchoMimic')
    gr.Markdown('![]()')
    with gr.Accordion("Configuration", open=False):
        width = gr.Slider(label="Width", minimum=128, maximum=1024, value=default_values["width"])
        height = gr.Slider(label="Height", minimum=128, maximum=1024, value=default_values["height"])
        length = gr.Slider(label="Length", minimum=100, maximum=5000, value=default_values["length"])
        seed = gr.Slider(label="Seed", minimum=0, maximum=10000, value=default_values["seed"])
        facemask_dilation_ratio = gr.Slider(label="Facemask Dilation Ratio", minimum=0.0, maximum=1.0, step=0.01, value=default_values["facemask_dilation_ratio"])
        facecrop_dilation_ratio = gr.Slider(label="Facecrop Dilation Ratio", minimum=0.0, maximum=1.0, step=0.01, value=default_values["facecrop_dilation_ratio"])
        context_frames = gr.Slider(label="Context Frames", minimum=0, maximum=50, step=1, value=default_values["context_frames"])
        context_overlap = gr.Slider(label="Context Overlap", minimum=0, maximum=10, step=1, value=default_values["context_overlap"])
        cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0, step=0.1, value=default_values["cfg"])
        steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=default_values["steps"])
        sample_rate = gr.Slider(label="Sample Rate", minimum=8000, maximum=48000, step=1000, value=default_values["sample_rate"])
        fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=default_values["fps"])
        device = gr.Radio(label="Device", choices=["cuda", "cpu"], value=default_values["device"])    
    with gr.Row():
        with gr.Column():
            uploaded_img = gr.Image(type="filepath", label="Reference Image")
            uploaded_audio = gr.Audio(type="filepath", label="Input Audio")
        with gr.Column():
            generate_button = gr.Button("Audio-Drived Algo")            
            output_video = gr.Video()
            generate_button_pose = gr.Button("Motion Alignment between Ref. Img. and Driven Vid")
            output_video_pose = gr.Video()
        
    generate_button.click(
        process_video,
        inputs=[
            uploaded_img,
            uploaded_audio,
            width,
            height,
            length,
            seed,
            facemask_dilation_ratio,
            facecrop_dilation_ratio,
            context_frames,
            context_overlap,
            cfg,
            steps,
            sample_rate,
            fps,
            device
        ],
        outputs=output_video
    )

    generate_button_pose.click(
        process_motion_sync,
        inputs=[
            output_video,
            uploaded_img,
            uploaded_audio,
            width,
            height,
            fps
        ],
        outputs=output_video_pose
    )

parser = argparse.ArgumentParser(description='EchoMimic')
parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
parser.add_argument('--server_port', type=int, default=7680, help='Server port')
args = parser.parse_args()

if __name__ == '__main__':
    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)

