"""Generic multimodal SFT collator for arbitrary vision-language models.

Unlike Qwen2VLCollator (which hunts a Qwen-specific assistant marker), this
collator masks the prompt by *length*: it tokenizes the prompt-only turn (with
the image expanded) to measure how many real tokens precede the assistant
response, then masks exactly those. This is model-agnostic and robust to the
tokenizer's padding side.
"""
from __future__ import annotations

from typing import Any

from PIL import Image


class GenericVLMCollator:
    """Collate unified records (image_path, user_text, assistant_text) for any VLM.

    Masks everything up to the assistant response with -100 so loss is computed
    only on the generated report.
    """

    def __init__(self, processor: Any, max_length: int = 2048) -> None:
        self.processor = processor
        self.max_length = max_length
        # Some processors (Mllama/Llama-3.2-Vision, Gemma3/MedGemma) require images
        # as a nested list (one sub-list per text sample); Qwen/InternVL take a flat list.
        _pname = type(processor).__name__.lower()
        self.nested_images = any(k in _pname for k in ("mllama", "gemma3"))

    def _messages(self, img: Image.Image, user_text: str, assistant_text: str | None):
        user = {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text},
        ]}
        if assistant_text is None:
            return [user]
        return [user, {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        full_texts, prompt_lens = [], []

        for r, img in zip(batch, images):
            msgs_full = self._messages(img, r["user_text"], r["assistant_text"])
            full_texts.append(self.processor.apply_chat_template(
                msgs_full, tokenize=False, add_generation_prompt=False))
            # prompt-only (with generation cue) → number of real tokens before response
            msgs_prompt = self._messages(img, r["user_text"], None)
            p = self.processor(
                text=[self.processor.apply_chat_template(
                    msgs_prompt, tokenize=False, add_generation_prompt=True)],
                images=[[img]] if self.nested_images else [img], return_tensors="pt",
            )
            prompt_lens.append(int(p["attention_mask"].sum().item()))

        batch_images = [[im] for im in images] if self.nested_images else images
        inputs = self.processor(
            text=full_texts, images=batch_images,
            return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()
        am = inputs["attention_mask"]
        for i in range(labels.shape[0]):
            real = am[i].nonzero(as_tuple=True)[0]      # positions of non-pad tokens, in order
            n_prompt = min(prompt_lens[i], real.numel())
            labels[i, real[:n_prompt]] = -100           # mask the prompt (padding-side robust)
        labels[am == 0] = -100                          # mask padding
        inputs["labels"] = labels
        return inputs
