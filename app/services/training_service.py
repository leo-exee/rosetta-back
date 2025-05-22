import logging
import platform

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


class ExerciseModelTrainer:
    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name

        # Detect the device: MPS for Mac M2, CUDA for GPU, else CPU
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Mac M2 GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

        # Add custom tokens for specific exercise formats
        special_tokens = [
            "<FILL_BLANK>",
            "<DEFINITION>",
            "<WORD>",
            "<CONTEXT>",
            "<QUESTION>",
            "<ANSWER>",
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_training_data(self, df: pd.DataFrame) -> Dataset:
        """Generate training dataset from various exercise types."""
        training_examples = []

        for _, row in df.iterrows():
            text = row["text"]

            # Fill-in-the-blank exercises
            training_examples.extend(self._create_fill_blank_examples(text))

            # Definition-based exercises
            if row["type"] == "vocabulary":
                training_examples.extend(self._create_definition_examples(text))

            # Reading comprehension exercises
            training_examples.extend(self._create_comprehension_examples(text))

        # Convert to HuggingFace Dataset and tokenize
        dataset = Dataset.from_pandas(pd.DataFrame(training_examples))
        return dataset.map(
            self._tokenize_function, batched=True, remove_columns=dataset.column_names
        )

    def _create_fill_blank_examples(self, text: str) -> list[dict]:
        """Create fill-in-the-blank exercises from sentences."""
        import random
        import re

        sentences = re.split(r"[.!?]+", text)
        examples = []

        for sentence in sentences[:3]:  # Limit to 3 sentences
            words = sentence.split()
            if len(words) < 5:
                continue

            if len(words) > 8:
                # Choose a content word (not an article/verb)
                content_words = [
                    i
                    for i, word in enumerate(words)
                    if word.lower()
                    not in ["the", "a", "an", "is", "are", "was", "were"]
                ]
                if content_words:
                    mask_idx = random.choice(content_words)
                    masked_word = words[mask_idx]

                    words[mask_idx] = "____"
                    input_text = f"fill_blank: {' '.join(words)}"
                    examples.append(
                        {
                            "input_text": input_text,
                            "target_text": masked_word,
                            "exercise_type": "fill_blank",
                        }
                    )

        return examples

    def _create_definition_examples(self, text: str) -> list[dict]:
        """Create definition and reverse-definition exercises from vocabulary entries."""
        if ":" in text:
            word, definition = text.split(":", 1)
            return [
                {
                    "input_text": f"define: {word.strip()}",
                    "target_text": definition.strip(),
                    "exercise_type": "definition",
                },
                {
                    "input_text": f"word_for_definition: {definition.strip()}",
                    "target_text": word.strip(),
                    "exercise_type": "reverse_definition",
                },
            ]
        return []

    def _create_comprehension_examples(self, text: str) -> list[dict]:
        """Generate basic comprehension questions based on named entities."""
        import re

        sentences = re.split(r"[.!?]+", text)
        examples = []

        for sentence in sentences[:2]:
            words = sentence.split()
            if len(words) > 10:
                nouns = [
                    word
                    for word in words
                    if word[0].isupper() and word not in ["The", "A", "An"]
                ]
                if nouns:
                    question = f"What is mentioned in: {sentence}?"
                    examples.append(
                        {
                            "input_text": f"comprehension: {question}",
                            "target_text": nouns[0],
                            "exercise_type": "comprehension",
                        }
                    )

        return examples

    def _tokenize_function(self, examples):
        """Tokenize input and target texts for model training."""
        inputs = list(examples["input_text"])
        targets = list(examples["target_text"])

        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True, padding=True
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, max_length=128, truncation=True, padding=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train_model(
        self, dataset: Dataset, output_dir: str = "data/models/exercise_model"
    ):
        """Train the model using the generated dataset."""

        # Split dataset
        train_test = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test["train"]
        eval_dataset = train_test["test"]

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4 if self.device.type == "cpu" else 8,
            per_device_eval_batch_size=4 if self.device.type == "cpu" else 8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb or tensorboard logging
            dataloader_pin_memory=False if self.device.type == "mps" else True,
            fp16=False if self.device.type in ["mps", "cpu"] else True,
        )

        # Define data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, padding=True
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")
        return trainer


async def train_model_service():
    df = pd.read_csv("data/raw/scraped_data.csv")
    trainer = ExerciseModelTrainer()
    logging.info("Preparing training data...")
    dataset = trainer.prepare_training_data(df)
    logging.info("Training model...")
    trainer.train_model(dataset)
    logging.info("Training completed.")
