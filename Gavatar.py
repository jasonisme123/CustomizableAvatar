
import cv2
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
import os


def generate_avatar(images_dir, prompt):
    app = FaceAnalysis(name="buffalo_l", providers=[
        'CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    images = []
    for root, dirs, files in os.walk(images_dir):
        for name in files:
            images.append(os.path.join(root, name))

    faceid_embeds = []
    for image in images:
        img = cv2.imread(image)
        faces = app.get(img)
        faceid_embeds.append(torch.from_numpy(
            faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
    faceid_embeds = torch.cat(faceid_embeds, dim=1)

    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    ip_ckpt = "ip-adapter-faceid-portrait_sd15.bin"
    device = "cpu"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    # load ip-adapter
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

    # generate image
    # prompt = "photo of a girl with a dark hair"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=1, width=512, height=512, num_inference_steps=30, seed=2023
    )
    images[0].save("avatar.png")


generate_avatar('./images', 'a girl with a red hair')
