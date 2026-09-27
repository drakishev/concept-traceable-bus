"""Data collator for Qwen2-VL multimodal SFT.

Handles image loading, processor calls, and label masking so that
only the assistant turns are trained on (user prompts get -100 labels).

Usage:
    from src.data.collator import Qwen2VLCollator
    collator = Qwen2VLCollator(processor, max_length=2048)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class Qwen2VLCollator:
    """Collates a batch of unified records into Qwen2-VL model inputs.

    Each record in the batch must have:
        image_path    : str path to the image file
        user_text     : user turn text (without <image> token)
        assistant_text: assistant turn text

    The collator:
    1. Opens each image.
    2. Builds the full conversation text via apply_chat_template.
    3. Calls processor(text, images) to get token ids + pixel values.
    4. Builds labels by copying input_ids and masking the prompt portion with -100
       so the model only learns to predict the assistant response.
    """

    def __init__(self, processor: Any, max_length: int = 2048) -> None:
        self.processor = processor
        self.max_length = max_length
        # Token id sequence that marks the start of the assistant turn:
        # "<|im_start|>assistant\n". We mask everything up to and including
        # this marker so loss is computed ONLY on the assistant response.
        # Computing prompt length from text-only tokenization is WRONG for
        # Qwen2-VL because the processor expands each image into hundreds of
        # <|image_pad|> tokens that text-only tokenization does not see.
        self._assistant_marker = processor.tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        texts: list[str] = []
        all_image_inputs: list = []

        for record in batch:
            img_path = record["image_path"]
            user_text = record["user_text"]
            assistant_text = record["assistant_text"]

            # Full conversation (user + assistant)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ]

            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Collect image inputs for this sample
            image_inputs, _ = process_vision_info(messages)
            all_image_inputs.extend(image_inputs)

            texts.append(full_text)

        # Process all texts + images together (expands image-pad tokens)
        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Build labels: copy input_ids, then mask everything except the
        # assistant response. We locate the assistant-turn marker token
        # subsequence in the *processed* input_ids (which already contains the
        # expanded image-pad tokens) and mask up to and including it.
        labels = inputs["input_ids"].clone()
        marker = torch.tensor(self._assistant_marker, dtype=labels.dtype)
        m = len(marker)
        for i in range(labels.shape[0]):
            ids = inputs["input_ids"][i]
            resp_start = None
            # search left-to-right for the LAST assistant marker (the response turn)
            for pos in range(ids.shape[0] - m, -1, -1):
                if torch.equal(ids[pos:pos + m], marker):
                    resp_start = pos + m
                    break
            if resp_start is None:
                # marker not found (truncated/malformed) → mask whole sample
                labels[i, :] = -100
            else:
                labels[i, :resp_start] = -100
        # Mask padding tokens
        labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        return inputs
