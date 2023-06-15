from math import e, log10
import sys
import os

from tqdm.auto import trange

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy

def progToMix(prog, pos, slope, shift = 0):
    return (1 / ( 1 + e**( -( prog + ( pos / ( -100 ) ) ) * slope ) )) + (shift / 100)

def progToSigma(prog, slope, sig_max, sig_min):
    return ((e ** (-((prog ** slope) - 2))) / e - 1) / (e - 1) * (sig_max - sig_min) + sig_min

class scaleSampler_:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "upscale_method": (["nearest-exact", "bilinear", "area"], ),
                    "pos" : ("FLOAT", {"default": 55.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                    "slope" : ("FLOAT", {"default": 16.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "scale_steps": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                    "maxScale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.2}),
                    "sps": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                    "sigma_start": ("FLOAT", {"default": 10.0, "min": 0.001, "max": 2000.0, "step": 0.1}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, noise_seed, upscale_method, pos, slope, cfg, sampler_name, scheduler, positive, negative, latent_image, scale_steps, maxScale, sps, sigma_start, denoise=1.0):
        step_size = (maxScale - 1) / scale_steps
        factors = []
        for i in range(0, scale_steps):
            factors.append(i * step_size + 1)


        device = comfy.model_management.get_torch_device()
        samples = latent_image["samples"]
        
        samples_list = [];
        for f in factors:
            item_size = (int(((samples.shape[2] * f) // 8) * 8), int(((samples.shape[3] * f) // 8) * 8))
            samples_list.append(torch.nn.functional.interpolate(samples, size=item_size, mode=upscale_method).to(device))

        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        result_noise = comfy.sample.prepare_noise(samples_list[-1], noise_seed, batch_inds)

        comfy.model_management.load_model_gpu(model)
        real_model = None or model.model

        positive_copy = comfy.sample.broadcast_cond(positive, result_noise.shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, result_noise.shape[0], device)

        models = comfy.sample.load_additional_models(positive, negative)

        samplers = []
        for i in factors:
            samplers.append(comfy.samplers.KSampler(real_model, steps=sps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options))
        #print(sampler.sigmas)
        #sigmas_orig = sampler.sigmas
        #sampler = comfy.samplers.KSampler(real_model, steps=1, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        #sampler.sigmas = sampler.sigmas * (sigma_start / sampler.sigmas[0])
        result_noise = result_noise.to(device)
        #for i, s in enumerate(samples_list):
        #    s += sampler.sigmas[0] * ([*reversed(factors)][i] / factors[-1]) * torch.nn.functional.interpolate(result_noise, size=(s.shape[2], s.shape[3]), mode=upscale_method)

        empty_result_noise = torch.zeros_like(result_noise)

        pbar = comfy.utils.ProgressBar(sps)
        sampled = []
        print(len(factors))
        # simple @ 17
        # 14.6094, 10.4219, 7.6016, 5.6875, 4.3555, 3.4023, 2.7148, 2.1875, 1.7842, 1.4648, 1.2070, 0.9941, 0.8154, 0.6572, 0.5156, 0.3821, 0.2461,  0.0000
        # @slope 0.2
        # 14.6094, 4.6770,  3.5840, 2.9179, 2.4351, 2.0555, 1.7426, 1.4763, 1.2445, 1.0394, 0.8554, 0.6886, 0.5362, 0.3958, 0.2657, 0.1446, 0.0312,
        # @slope 0.6
        # 14.6094, 10.5987, 8.8215, 7.5071, 6.4433, 5.5439, 4.7628, 4.0721, 3.4534, 2.8936, 2.3830, 1.9142, 1.4815, 1.0803, 0.7069, 0.3581, 0.0312,


        for j in range(0, sps):
            for i, f in enumerate(factors):
            #@todo not done
                #prog = ((i - 1) * (j + 1) + (j + 1)) / (len(factors) * sps - 1)
                #print(prog)
                #prev_sigma = sampler.sigmas[0]
                #sig_factor = progToSigma(prog, slope, sampler.sigma_max, sampler.sigma_min)

                if j == 0:
                    #sig_factor = sigmas_orig[i]
                    #sampler.sigmas[0] = sigmas_orig[i]
                    #sampler.sigmas[0] = sig_factor
                    #print(sig_factor)
                    s = samples_list[i]
                    #s += sampler.sigmas[0] * torch.nn.functional.interpolate(result_noise, size=(s.shape[2], s.shape[3]), mode=upscale_method)
                    s += torch.nn.functional.interpolate(result_noise, size=(s.shape[2], s.shape[3]), mode=upscale_method)
                elif sampled:
                    diff = sampled[i-1] - torch.nn.functional.interpolate(samples_list[i], size=(sampled[i-1].shape[2], sampled[i-1].shape[3]), mode=upscale_method)
                    samples_list[i] = samples_list[i] + (torch.nn.functional.interpolate(diff, size=(samples_list[i].shape[2], samples_list[i].shape[3]), mode=upscale_method))

                empty_noise = torch.nn.functional.interpolate(empty_result_noise, size=(samples_list[i].shape[2], samples_list[i].shape[3]), mode=upscale_method)
                sampled.append(samplers[i].sample(
                    empty_noise,
                    positive_copy,
                    negative_copy,
                    cfg=cfg,
                    latent_image=samples_list[i],
                    start_step=j,
                    last_step=sps,
                    force_full_denoise=False,
                    denoise_mask=None,
                    disable_pbar=True
                ))

            pbar.update_absolute(j + 1, sps)
            
        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = sampled[-1].cpu()
        return (out, )


class scaleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "pos" : ("FLOAT", {"default": 50.0, "min": -200, "max": 200.0, "step": 0.1}),
                    "slope" : ("FLOAT", {"default": 4.0, "min": -50, "max": 50.0, "step": 0.1}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "scale_steps": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                    "maxScale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 32.0, "step": 0.2}),
                    "cfg_scale": ("FLOAT", {"default": 1, "min": -30.0, "max": 30.0, "step": 0.1}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, noise_seed, pos, slope, cfg, sampler_name, scheduler, positive, negative, latent_image, scale_steps, maxScale, cfg_scale, denoise=1.0):
        slope_min = progToMix(0, pos, slope)
        slope_max = progToMix(1, pos, slope)
        factors = []
        for i in range(0, scale_steps):
            factors.append(1 + ((progToMix(i / (scale_steps - 1), pos, slope) - slope_min) / (slope_max - slope_min)) * (maxScale - 1))
        #print(factors)

        device = comfy.model_management.get_torch_device()
        samples = latent_image["samples"]
        
        samples_list = [];
        for f in factors:
            item_size = (int(((samples.shape[2] * f) // 8) * 8), int(((samples.shape[3] * f) // 8) * 8))
            samples_list.append(torch.nn.functional.interpolate(samples, size=item_size, mode="bilinear").to(device))

        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        result_noise = comfy.sample.prepare_noise(samples_list[-1], noise_seed, batch_inds)

        comfy.model_management.load_model_gpu(model)
        real_model = None or model.model

        positive_copy = comfy.sample.broadcast_cond(positive, result_noise.shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, result_noise.shape[0], device)

        models = comfy.sample.load_additional_models(positive, negative)

        sampler = comfy.samplers.KSampler(real_model, steps=len(factors), device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        sigmas_orig = sampler.sigmas
        sampler = comfy.samplers.KSampler(real_model, steps=1, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        result_noise = result_noise.to(device)

        empty_result_noise = torch.zeros_like(result_noise)

        pbar = comfy.utils.ProgressBar(len(factors))
        sampled = []
        for i, f in enumerate(factors):
            prog = i / (len(factors) - 1)
            prev_sigma = sampler.sigmas[0]
            cfg_fact = (progToMix(prog, pos, slope) - 0.5) * cfg_scale

            sig_factor = sigmas_orig[i]
            sampler.sigmas[0] = sig_factor

            if sampled:
                diff = sampled[-1] / prev_sigma - torch.nn.functional.interpolate(result_noise, size=(sampled[-1].shape[2], sampled[-1].shape[3]), mode="bilinear")
                result_noise = result_noise + torch.nn.functional.interpolate(diff, size=(result_noise.shape[2], result_noise.shape[3]), mode="bilinear")
                diff_mean = torch.mean(torch.abs(diff)).item()

            #samples_list[i] = torch.nn.functional.interpolate(result_noise + comfy.sample.prepare_noise(result_noise, noise_seed + 1 + i, batch_inds).to(device) * (sig_factor / sigmas_orig[0]), size=(samples_list[i].shape[2], samples_list[i].shape[3]), mode="bilinear") * sig_factor
            samples_list[i] = torch.nn.functional.interpolate(
                result_noise + comfy.sample.prepare_noise(result_noise, noise_seed + 1 + i, batch_inds).to(device) * ((0 if not sampled else diff_mean) * (sig_factor / sigmas_orig[0])),
                #result_noise + comfy.sample.prepare_noise(result_noise, noise_seed + 1 + i, batch_inds).to(device) * ((sig_factor / sigmas_orig[0])),
                size=(samples_list[i].shape[2], samples_list[i].shape[3]),
                mode="bilinear"
            ) * sig_factor
            samples_list[i] = samples_list[i] + comfy.sample.prepare_noise(samples_list[i], noise_seed + 7 + i, batch_inds).to(device) * (sig_factor) 


            empty_noise = torch.nn.functional.interpolate(empty_result_noise, size=(samples_list[i].shape[2], samples_list[i].shape[3]), mode="bilinear")
            #DEV before = samples_list[i].clone().detach()
            ccfg=torch.tensor((cfg + cfg * cfg_fact))
            sampler.sigmas = sigmas_orig.clone()
            sampled.append(sampler.sample(
                empty_noise,
                positive_copy,
                negative_copy,
                cfg=ccfg,
                latent_image=samples_list[i],
                start_step=i,
                last_step=len(factors),
                force_full_denoise=False,
                denoise_mask=None,
                disable_pbar=True
            ))
            #DEV removed_by_sampling = torch.mean(torch.abs(before - sampled[-1])).item()
            #DEV print('s', sig_factor, 'ccfg', ccfg.item(), 'mean', removed_by_sampling, 'dif', 0 if len(sampled) == 1 else diff_mean, (cfg_fact, cfg + cfg * cfg_fact))

            pbar.update_absolute(i + 1, len(factors))
            
        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = sampled[-1].cpu()
        return (out, )

class guidedSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "upscale_method": (["nearest-exact", "bilinear", "area"], ),
                    "pos" : ("FLOAT", {"default": 55.0, "min": -200.0, "max": 1200.0, "step": 0.5}),
                    "slope" : ("FLOAT", {"default": 16.0, "min": -200.0, "max": 1200.0, "step": 0.5}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "guide": ("LATENT", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, noise_seed, upscale_method, pos, slope, steps, cfg, sampler_name, scheduler, positive, negative, guide, latent_image, start_at_step, end_at_step, denoise=1.0):
        end_at_step = min(end_at_step, steps + start_at_step)
        device = comfy.model_management.get_torch_device()
        samples = latent_image["samples"]

        guide_samples = guide["samples"]
        guide_size = (guide_samples.shape[2], guide_samples.shape[3])
        size = (samples.shape[2], samples.shape[3])
        
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        result_noise = comfy.sample.prepare_noise(samples, noise_seed, batch_inds)
        

        comfy.model_management.load_model_gpu(model)
        real_model = None or model.model

        positive_copy = comfy.sample.broadcast_cond(positive, result_noise.shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, result_noise.shape[0], device)

        models = comfy.sample.load_additional_models(positive, negative)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        result_noise = result_noise.to(device)
        samples = samples.to(device)
        guide_samples = guide_samples.to(device)
        samples += sampler.sigmas[start_at_step] * result_noise

        empty_result_noise = torch.zeros_like(result_noise)

        pbar = comfy.utils.ProgressBar(steps)
        for i in trange(end_at_step - start_at_step):
            mix_factor = i / ( end_at_step - start_at_step-1 )
            mix_factor = progToMix(mix_factor, pos, slope)
            
            diff = guide_samples - torch.nn.functional.interpolate(samples, size=guide_size, mode=upscale_method)
            samples = samples + (torch.nn.functional.interpolate(diff, size=size, mode=upscale_method) * (1-mix_factor))
            samples = sampler.sample(empty_result_noise, positive_copy, negative_copy, cfg=cfg, latent_image=samples, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            pbar.update_absolute(i + 1, steps)
            
        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = samples.cpu()
        return (out, )


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
        
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise_up = comfy.sample.prepare_noise(samples_up, noise_seed, batch_inds)
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

        pbar = comfy.utils.ProgressBar(steps)
        for i in trange(end_at_step - start_at_step):
            mix_factor = i / ( end_at_step - start_at_step-1 )
            mix_factor = progToMix(mix_factor, pos, slope)
            
            samples = sampler.sample(empty_noise, positive_copy, negative_copy, cfg=cfg, latent_image=samples, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            diff = samples - torch.nn.functional.interpolate(samples_up, size=size_small, mode=downsample)
            samples_up = samples_up + (torch.nn.functional.interpolate(diff, size=size_large, mode=upsample) * (1-mix_factor))
            samples_up = sampler.sample(empty_noise_up, positive_copy, negative_copy, cfg=cfg, latent_image=samples_up, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            pbar.update_absolute(i + 1, steps)
            
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
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise_xl = comfy.sample.prepare_noise(samples_xl, noise_seed, batch_inds)
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

        pbar = comfy.utils.ProgressBar(steps)
        for i in trange(end_at_step - start_at_step):
            prog = i / (end_at_step - start_at_step-1)
            mix_factor = progToMix(prog, pos, slope)

            samples = sampler.sample(empty_noise, positive_copy, negative_copy, cfg=cfg, latent_image=samples, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            
            diff = samples - torch.nn.functional.interpolate(samples_up, size=size_small, mode=upscale_method)
            samples_up = samples_up + (torch.nn.functional.interpolate(diff, size=size_large, mode=upscale_method) * (1-mix_factor))

            samples_up = sampler.sample(empty_noise_up, positive_copy, negative_copy, cfg=cfg, latent_image=samples_up, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            
            diff = samples_up - torch.nn.functional.interpolate(samples_xl, size=size_large, mode=upscale_method)
            samples_xl = samples_xl + (torch.nn.functional.interpolate(diff, size=size_xl, mode=upscale_method) * (1-(mix_factor)))
            
            samples_xl = sampler.sample(empty_noise_xl, positive_copy, negative_copy, cfg=cfg, latent_image=samples_xl, start_step=start_at_step + i, last_step=start_at_step + i + 1, force_full_denoise=False, denoise_mask=None, disable_pbar=True)
            pbar.update_absolute(i + 1, steps)
            
        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = samples_xl.cpu()
        return (out, )


    
NODE_CLASS_MAPPINGS = {
    "x2Sampler": x2Sampler,
    "x4Sampler": x4Sampler,
    "guidedSampler": guidedSampler,
    "scaleSampler": scaleSampler,
}
