from math import e
import sys
import os

from tqdm.auto import trange

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy

class x2Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "upscale_method": (["nearest-exact", "bilinear", "area"], ),
                    "pos" : ("FLOAT", {"default": 55.0, "min": -20.0, "max": 120.0, "step": 0.5}),
                    "slope" : ("FLOAT", {"default": 16.0, "min": -20.0, "max": 120.0, "step": 0.5}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, noise_seed, upscale_method, pos, slope, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, denoise=1.0):
        downsample = upscale_method
        upsample = upscale_method
        end_at_step = min(end_at_step, steps + start_at_step)
        device = comfy.model_management.get_torch_device()
        samples = latent_image["samples"]

        size_small = (samples.shape[2], samples.shape[3])
        size_large = (int(((samples.shape[2] * 2) // 8) * 8), int(((samples.shape[3] * 2) // 8) * 8))

        samples_up = torch.nn.functional.interpolate(samples, size=size_large, mode=upsample)
        skip = latent_image["batch_index"] if "batch_index" in latent_image else 0
        noise_up = comfy.sample.prepare_noise(samples_up, noise_seed, skip)
        noise = torch.nn.functional.interpolate(noise_up, size=size_small, mode=downsample)
        

        comfy.model_management.load_model_gpu(model)
        real_model = None or model.model

        positive_copy = comfy.sample.broadcast_cond(positive, noise.shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, noise.shape[0], device)

        models = comfy.sample.load_additional_models(positive, negative)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        noise = noise.to(device)
        noise_up = noise_up.to(device)
        samples = samples.to(device)
        samples_up = samples_up.to(device)
        samples += sampler.sigmas[start_at_step] * noise
        samples_up += sampler.sigmas[start_at_step] * noise_up

        empty_noise_up = torch.zeros_like(noise_up)
        empty_noise = torch.zeros_like(noise)

        for i in trange(end_at_step - start_at_step):
            mix_factor = i / ( end_at_step - start_at_step-1 )
            mix_factor = 1 / ( 1 + e**( -( mix_factor + ( pos / ( -100 ) ) ) * slope ) )
            
            samples = sampler.sample(empty_noise, positive_copy, negative_copy, cfg=cfg, latent_image=samples, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            diff = samples - torch.nn.functional.interpolate(samples_up, size=size_small, mode=downsample)
            samples_up = samples_up + (torch.nn.functional.interpolate(diff, size=size_large, mode=upsample) * (1-mix_factor))
            samples_up = sampler.sample(empty_noise_up, positive_copy, negative_copy, cfg=cfg, latent_image=samples_up, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=False)
            
        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = samples_up.cpu()
        return (out, )


class x4Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "upscale_method": (["nearest-exact", "bilinear", "area"], ),
                    "pos" : ("FLOAT", {"default": 55.0, "min": -20.0, "max": 120.0, "step": 0.5}),
                    "slope" : ("FLOAT", {"default": 16.0, "min": -20.0, "max": 120.0, "step": 0.5}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, noise_seed, upscale_method, pos, slope, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, denoise=1.0):
        end_at_step = min(end_at_step, steps + start_at_step)
        device = comfy.model_management.get_torch_device()
        samples = latent_image["samples"]

        size_small = (samples.shape[2], samples.shape[3])
        size_large = (int(((samples.shape[2] * 2) // 8) * 8), int(((samples.shape[3] * 2) // 8) * 8))
        size_xl = (int(((samples.shape[2] * 4) // 8) * 8), int(((samples.shape[3] * 4) // 8) * 8))

        samples_up = torch.nn.functional.interpolate(samples, size=size_large, mode=upscale_method)
        samples_xl = torch.nn.functional.interpolate(samples, size=size_xl, mode=upscale_method)
        skip = latent_image["batch_index"] if "batch_index" in latent_image else 0
        noise_xl = comfy.sample.prepare_noise(samples_xl, noise_seed, skip)
        noise_up = torch.nn.functional.interpolate(noise_xl, size=size_large, mode=upscale_method)
        noise = torch.nn.functional.interpolate(noise_up, size=size_small, mode=upscale_method)
        

        comfy.model_management.load_model_gpu(model)
        real_model = None or model.model

        positive_copy = comfy.sample.broadcast_cond(positive, noise.shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, noise.shape[0], device)

        models = comfy.sample.load_additional_models(positive, negative)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        noise = noise.to(device)
        noise_up = noise_up.to(device)
        noise_xl = noise_xl.to(device)
        samples = samples.to(device)
        samples_up = samples_up.to(device)
        samples_xl = samples_xl.to(device)
        samples += sampler.sigmas[start_at_step] * noise
        samples_up += sampler.sigmas[start_at_step] * noise_up
        samples_xl += sampler.sigmas[start_at_step] * noise_xl

        empty_noise_xl = torch.zeros_like(noise_xl)
        empty_noise_up = torch.zeros_like(noise_up)
        empty_noise = torch.zeros_like(noise)

        for i in trange(end_at_step - start_at_step):
            prog = i / (end_at_step - start_at_step-1)
            mix_factor = 1 / ( 1 + e ** ( -(prog + pos / (-100) ) * slope ) )

            samples = sampler.sample(empty_noise, positive_copy, negative_copy, cfg=cfg, latent_image=samples, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            
            diff = samples - torch.nn.functional.interpolate(samples_up, size=size_small, mode=upscale_method)
            samples_up = samples_up + (torch.nn.functional.interpolate(diff, size=size_large, mode=upscale_method) * (1-mix_factor))

            samples_up = sampler.sample(empty_noise_up, positive_copy, negative_copy, cfg=cfg, latent_image=samples_up, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            
            diff = samples_up - torch.nn.functional.interpolate(samples_xl, size=size_large, mode=upscale_method)
            samples_xl = samples_xl + (torch.nn.functional.interpolate(diff, size=size_xl, mode=upscale_method) * (1-(mix_factor)))
            
            samples_xl = sampler.sample(empty_noise_xl, positive_copy, negative_copy, cfg=cfg, latent_image=samples_xl, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=False)
            
        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = samples_xl.cpu()
        return (out, )

    
NODE_CLASS_MAPPINGS = {
    "x2Sampler": x2Sampler,
    "x4Sampler": x4Sampler,
}
