def load_model(model_path):
    import torch
    from diffusers import StableDiffusionPipeline

    model = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to("cuda")
    return model

def generate_image(prompt, model, num_inference_steps=50, guidance_scale=7.5):
    with torch.no_grad():
        images = model(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
    return images

def save_image(image, output_path):
    image.save(output_path)

def run_inference(prompt, model_path, output_path):
    model = load_model(model_path)
    images = generate_image(prompt, model)
    for i, img in enumerate(images):
        save_image(img, f"{output_path}/generated_image_{i}.png")