"""Small translation helpers shared by Stage 2 corpus builders."""

from __future__ import annotations

import torch


NLLB_ALIASES = {
    "cmn_Hans": "zho_Hans",
    "cmn_Hant": "zho_Hant",
}


class NllbSentenceTranslator:
    """Translate independent English sentences with an NLLB-compatible model."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        max_new_tokens: int,
        local_files_only: bool,
        revision: str | None = None,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            revision=revision,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            revision=revision,
        ).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer.src_lang = "eng_Latn"
        self.vocabulary = set(self.tokenizer.get_vocab())
        self.model.generation_config.max_length = None

    def supports(self, language: str) -> bool:
        return NLLB_ALIASES.get(language, language) in self.vocabulary

    def __call__(self, sentences: list[str], target_language: str) -> list[str]:
        target_language = NLLB_ALIASES.get(target_language, target_language)
        if not self.supports(target_language):
            raise ValueError(f"{target_language!r} is unsupported by the MT tokenizer.")
        print(f"translating {target_language}: {len(sentences)} sentences", flush=True)
        target_id = self.tokenizer.convert_tokens_to_ids(target_language)
        translations = []
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=target_id,
                    max_new_tokens=self.max_new_tokens,
                )
            translations.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return [translation.strip() for translation in translations]
