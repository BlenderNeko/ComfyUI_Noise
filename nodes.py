import torch

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.model_management
import comfy.sample

MAX_RESOLUTION=8192

class NoisyLatentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "source":(["CPU", "GPU"], ),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "create_noisy_latents"

    CATEGORY = "latent/noise"
        
    def create_noisy_latents(self, source, seed, width, height, batch_size):
        torch.manual_seed(seed)
        if source == "CPU":
            device = "cpu"
        else:
            device = comfy.model_management.get_torch_device()
        noise = torch.randn((batch_size,  4, height // 8, width // 8), dtype=torch.float32, device=device).cpu()
        return ({"samples":noise}, )

class DuplicateBatchIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents":("LATENT",),
            "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "duplicate_index"

    CATEGORY = "latent"
        
    def duplicate_index(self, latents, batch_index, batch_size):
        s = latents.copy()
        batch_index = min(s["samples"].shape[0] - 1, batch_index)
        target = s["samples"][batch_index:batch_index + 1].clone()
        target = target.repeat((batch_size,1,1,1))
        s["samples"] = target
        return (s,)

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp(val, low, high):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)

class LatentSlerp:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents1":("LATENT",),
            "latents2":("LATENT",),
            "factor": ("FLOAT", {"default": .5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "slerp_latents"

    CATEGORY = "latent"
        
    def slerp_latents(self, latents1, latents2, factor):
        s = latents1.copy()
        if latents1["samples"].shape != latents2["samples"].shape:
            print("warning, shapes in LatentSlerp not the same, ignoring")
            return (s,)
        s["samples"] = slerp(factor, latents1["samples"].clone(), latents2["samples"].clone())
        return (s,)

class GetSigma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            "steps": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            }}
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calc_sigma"

    CATEGORY = "latent/noise"
        
    def calc_sigma(self, model, sampler_name, scheduler, steps, start_at_step, end_at_step):
        device = comfy.model_management.get_torch_device()
        end_at_step = min(steps, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model
        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        return (sigma.cpu().numpy(),)

class InjectNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents":("LATENT",),
            "noise":  ("LATENT",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject_noise"

    CATEGORY = "latent/noise"
        
    def inject_noise(self, latents, noise, strength):
        s = latents.copy()
        if latents["samples"].shape != noise["samples"].shape:
            print("warning, shapes in InjectNoise not the same, ignoring")
            return (s,)
        s["samples"] = s["samples"].clone() + noise["samples"].clone() * strength
        return (s,)
    
class Unsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "unsampler"

    CATEGORY = "sampling"
        
    def unsampler(self, model, cfg, sampler_name, steps, scheduler, positive, negative, latent_image):
        device = comfy.model_management.get_torch_device()
        latent = latent_image
        latent_image = latent["samples"]
        
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(latent["noise_mask"], noise, device)

        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        positive_copy = comfy.sample.broadcast_cond(positive, noise.shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, noise.shape[0], device)

        models = comfy.sample.load_additional_models(positive, negative)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sigmas = sampler.sigmas.flip(0)[1:] + 0.0001

        samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas)
        samples = samples.cpu()
        
        comfy.sample.cleanup_additional_models(models)

        out = latent.copy()
        out["samples"] = samples
        return (out, )
    
NODE_CLASS_MAPPINGS = {
    "BNK_NoisyLatentImage": NoisyLatentImage,
    "BNK_DuplicateBatchIndex": DuplicateBatchIndex,
    "BNK_SlerpLatent": LatentSlerp,
    "BNK_GetSigma": GetSigma,
    "BNK_InjectNoise": InjectNoise,
    "BNK_Unsampler": Unsampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BNK_NoisyLatentImage": "Noisy Latent Image",
    "BNK_DuplicateBatchIndex": "Duplicate Batch Index",
    "BNK_SlerpLatent": "Slerp Latents",
    "BNK_GetSigma": "Get Sigma",
    "BNK_InjectNoise": "Inject Noise",
    "BNK_Unsampler": "Unsampler",
}
