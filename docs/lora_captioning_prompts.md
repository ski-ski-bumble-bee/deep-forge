# LoRA Captioning Prompt Templates

## Style / Landscape LoRA
*(describe everything)*

```
You are captioning images for training an AI image generation model (style LoRA). 
Describe the full scene in 2-5 natural language sentences. Include: subject matter, 
lighting, atmosphere, colors, textures, composition, time of day. Do NOT reference 
artistic style, medium, or any photographer/artist name. Be precise and factual.
```

---

## Character LoRA
*(omit character appearance)*

```
You are captioning images for training a character LoRA. The character's appearance 
must NOT be described — treat them as a known entity referenced only by their trigger 
word "[TRIGGER]". Describe everything else: background/setting, pose, clothing worn 
by the character, facial expression if readable, lighting, camera angle, other people 
or objects in the scene. Write 2-4 natural sentences.
```

---

## Product LoRA
*(omit the product)*

```
You are captioning images for training a product LoRA. Do NOT describe the product 
itself — it will be referenced only as "[TRIGGER]". Describe: the model/person wearing 
or holding it (vary descriptors), their other clothing, the background/setting, lighting, 
camera angle, and pose. Include the product's color as an adjective before the trigger 
word. Write 2-4 natural sentences. Example: "A woman is wearing red [TRIGGER]. She 
stands in a modern kitchen, also wearing a white crop top. The photo is taken from 
the front at eye level with soft natural lighting."
```

---

## General-Purpose Auto-Captioner
*(with post-processing intent)*

```
Describe this image in 2-5 natural language sentences. Include the subjects, their 
positions and poses, clothing, the setting/background, lighting conditions, and camera 
angle. Do not mention artistic style, image quality, or aesthetic judgments. Do not 
reference skin tone. Use correct gender pronouns. Keep it factual and concise.
```
