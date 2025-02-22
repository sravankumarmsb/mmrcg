from diffusers import StableDiffusionPipeline
import torch

def captionToImage(caption) :
  # Load the pre-trained Stable Diffusion model
  model_id = "stabilityai/stable-diffusion-2"
  pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
  #pipe.to("cuda")  # Use GPU for faster generation (if available)
  # Define the caption (text prompt)
  #caption = "Two dogs of different breeds looking at each other on the road"
  # Generate Image
  image = pipe(caption).images[0]
  return image


BLEU2,BLEU3,BLEU4,WER,ROUGE-1

      val = str(row['Tokens']).strip()
