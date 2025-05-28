import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset, DatasetDict

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedAIModelsTrainer:
    """
    Entraîneur optimisé pour les trois modèles d'IA avec améliorations de performances
    """

    def __init__(self, base_output_dir: str = "models/trained", use_gpu: bool = True):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration GPU/CPU optimisée
        self.device = self._setup_device(use_gpu)
        self.use_mixed_precision = torch.cuda.is_available() and use_gpu

        # Configuration des modèles optimisée
        self.model_configs = {
            "fill_in_blank": {
                "model_name": "google/flan-t5-small",  # Gardé petit pour la vitesse
                "type": "seq2seq",
                "task_prefix": "Fill in the blanks: ",
                "max_input_length": 128,  # Réduit de 256 -> 128
                "max_target_length": 64,  # Réduit de 128 -> 64
                "batch_size": self._get_optimal_batch_size("seq2seq"),
            },
            "sentence_scrambler": {
                "model_name": "google/flan-t5-small",
                "type": "seq2seq",
                "task_prefix": "Unscramble sentence: ",
                "max_input_length": 96,  # Réduit de 128 -> 96
                "max_target_length": 64,  # Réduit de 128 -> 64
                "batch_size": self._get_optimal_batch_size("seq2seq"),
            },
            "definition_matcher": {
                "model_name": "distilbert-base-uncased",
                "type": "classification",
                "num_labels": 3,
                "max_input_length": 128,  # Réduit de 256 -> 128
                "batch_size": self._get_optimal_batch_size("classification"),
            },
        }

        # Cache pour les tokenizers
        self._tokenizer_cache = {}

        # Statistiques d'entraînement
        self.training_stats = {}

    def _setup_device(self, use_gpu: bool):
        """Configure le device optimal"""
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🔥 GPU détecté: {torch.cuda.get_device_name()}")
            # Optimisations CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = torch.device("cpu")
            logger.info("💻 Entraînement sur CPU")
            # Optimisations CPU
            torch.set_num_threads(min(8, os.cpu_count()))

        return device

    def _get_optimal_batch_size(self, model_type: str) -> dict:
        """Calcule les tailles de batch optimales selon le hardware"""
        if torch.cuda.is_available():
            # GPU - batch sizes plus grands
            if model_type == "seq2seq":
                return {"train": 8, "eval": 16}
            else:  # classification
                return {"train": 16, "eval": 32}
        else:
            # CPU - batch sizes plus petits
            if model_type == "seq2seq":
                return {"train": 4, "eval": 8}
            else:
                return {"train": 8, "eval": 16}

    def _get_cached_tokenizer(self, model_name: str):
        """Cache des tokenizers pour éviter les rechargements"""
        if model_name not in self._tokenizer_cache:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer_cache[model_name] = tokenizer
        return self._tokenizer_cache[model_name]

    def load_datasets_parallel(
        self, dataset_dir: str = "datasets/training_fr"
    ) -> dict[str, DatasetDict]:
        """Charge les datasets en parallèle pour plus de vitesse"""
        logger.info("📚 Chargement parallèle des datasets...")

        datasets = {}
        dataset_path = Path(dataset_dir)

        # Debug : vérifier que le dossier existe
        logger.info(f"🔍 Recherche dans : {dataset_path.absolute()}")
        if not dataset_path.exists():
            logger.error(f"❌ Dossier datasets non trouvé : {dataset_path}")
            return {}

        # Debug : lister les dossiers disponibles
        available_dirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        logger.info(f"📁 Dossiers disponibles : {available_dirs}")

        def load_single_dataset(model_name):
            model_dir = dataset_path / model_name
            logger.info(f"🔍 Chargement de {model_name} depuis {model_dir}")

            if not model_dir.exists():
                logger.warning(f"⚠️ Dossier non trouvé: {model_dir}")
                return model_name, None

            splits = {}
            for split in ["train", "val", "test"]:
                split_file = model_dir / f"{split}.jsonl"
                if split_file.exists():
                    with open(split_file, encoding="utf-8") as f:
                        data = [json.loads(line) for line in f if line.strip()]
                    splits[split] = data
                    logger.info(f"✅ {model_name}/{split}: {len(data)} exemples")
                else:
                    logger.warning(f"⚠️ Fichier manquant: {split_file}")

            return model_name, splits if splits else None

        # Chargement parallèle
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(load_single_dataset, model_name)
                for model_name in self.model_configs.keys()
            ]

            for future in as_completed(futures):
                model_name, data = future.result()
                if data:
                    datasets[model_name] = data
                    total_samples = sum(len(split_data) for split_data in data.values())
                    logger.info(f"✅ {model_name}: {total_samples} exemples totaux")
                else:
                    logger.warning(f"⚠️ Aucune donnée chargée pour {model_name}")

        logger.info(f"📊 Datasets chargés : {list(datasets.keys())}")
        return datasets

    def prepare_data_optimized(self, data: list[dict], model_name: str) -> Dataset:
        """Version optimisée de la préparation des données avec validation"""
        if model_name == "fill_in_blank":
            return self._prepare_fill_in_blank_optimized(data)
        elif model_name == "sentence_scrambler":
            return self._prepare_sentence_scrambler_optimized(data)
        elif model_name == "definition_matcher":
            return self._prepare_definition_matcher_optimized(data)
        else:
            raise ValueError(f"Modèle non reconnu: {model_name}")

    def _prepare_fill_in_blank_optimized(self, data: list[dict]) -> Dataset:
        """Version optimisée avec validation des données"""
        inputs, targets = [], []
        skipped = 0

        for item in data:
            try:
                input_parts = item["input"].split("|")
                output_parts = item["output"].split("|||")

                if len(input_parts) != 3 or len(output_parts) != 3:
                    skipped += 1
                    continue

                context, level, _ = input_parts
                text_with_blanks, words_to_fill, complete_text = output_parts

                # Input plus concis
                model_input = f"Context: {context[:50]}, Level: {level}. Fill: {text_with_blanks}. Options: {words_to_fill}"
                inputs.append(model_input)
                targets.append(complete_text)

            except Exception:
                skipped += 1
                continue

        if skipped > 0:
            logger.warning(f"⚠️ {skipped} exemples ignorés (format invalide)")

        return Dataset.from_dict({"input_text": inputs, "target_text": targets})

    def _prepare_sentence_scrambler_optimized(self, data: list[dict]) -> Dataset:
        """Version optimisée pour sentence scrambler"""
        inputs, targets = [], []
        skipped = 0

        for item in data:
            try:
                input_parts = item["input"].split("|")
                output_parts = item["output"].split("|||")

                if len(input_parts) != 3 or len(output_parts) != 2:
                    skipped += 1
                    continue

                context, level, _ = input_parts
                scrambled_words, target_sentence = output_parts

                # Input plus concis
                model_input = f"Context: {context[:50]}, Level: {level}. Unscramble: {scrambled_words}"
                inputs.append(model_input)
                targets.append(target_sentence)

            except Exception:
                skipped += 1
                continue

        if skipped > 0:
            logger.warning(f"⚠️ {skipped} exemples ignorés (format invalide)")

        return Dataset.from_dict({"input_text": inputs, "target_text": targets})

    def _prepare_definition_matcher_optimized(self, data: list[dict]) -> Dataset:
        """Version optimisée pour definition matcher"""
        inputs, labels = [], []
        skipped = 0

        for item in data:
            try:
                input_parts = item["input"].split("|")
                output_parts = item["output"].split("|||")

                if len(input_parts) != 2 or len(output_parts) != 5:
                    skipped += 1
                    continue

                context, level = input_parts
                words_str, def1, def2, def3, correct_matches = output_parts

                words = words_str.split(",")
                definitions = [def1, def2, def3]
                matches = [int(x) - 1 for x in correct_matches.split(",")]

                for word, correct_def_idx in zip(words, matches, strict=True):
                    # Input plus concis avec troncature
                    context_short = context[:30]
                    defs_short = " | ".join(
                        [d[:40] + "..." if len(d) > 40 else d for d in definitions]
                    )
                    model_input = f"Context: {context_short}, Level: {level}. Word: {word}. Defs: {defs_short}"

                    inputs.append(model_input)
                    labels.append(correct_def_idx)

            except Exception:
                skipped += 1
                continue

        if skipped > 0:
            logger.warning(f"⚠️ {skipped} exemples ignorés (format invalide)")

        return Dataset.from_dict({"input_text": inputs, "labels": labels})

    def tokenize_with_caching(
        self, dataset: Dataset, model_name: str, config: dict
    ) -> Dataset:
        """Tokenisation avec cache et optimisations"""
        tokenizer = self._get_cached_tokenizer(config["model_name"])

        if config["type"] == "seq2seq":
            return self._tokenize_seq2seq_optimized(dataset, tokenizer, config)
        else:
            return self._tokenize_classification_optimized(dataset, tokenizer, config)

    def _tokenize_seq2seq_optimized(
        self, dataset: Dataset, tokenizer, config: dict
    ) -> Dataset:
        """Tokenisation seq2seq optimisée"""

        def tokenize_function(examples):
            model_inputs = tokenizer(
                [config["task_prefix"] + text for text in examples["input_text"]],
                max_length=config["max_input_length"],
                truncation=True,
                padding="max_length",
                return_tensors=None,  # Évite la conversion automatique
            )

            targets = tokenizer(
                examples["target_text"],
                max_length=config["max_target_length"],
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )

            model_inputs["labels"] = targets["input_ids"]
            return model_inputs

        return dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,  # Batch plus grand pour la tokenisation
            num_proc=min(4, os.cpu_count()),  # Parallélisation
            remove_columns=dataset.column_names,
        )

    def _tokenize_classification_optimized(
        self, dataset: Dataset, tokenizer, config: dict
    ) -> Dataset:
        """Tokenisation classification optimisée"""

        def tokenize_function(examples):
            return tokenizer(
                examples["input_text"],
                max_length=config["max_input_length"],
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )

        return dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=min(4, os.cpu_count()),
            remove_columns=["input_text"],  # Garder labels pour classification
        )

    def get_optimized_training_args(self, model_name: str, model_type: str) -> dict:
        """Arguments d'entraînement optimisés"""
        config = self.model_configs[model_name]
        batch_size = config["batch_size"]

        base_args = {
            "output_dir": str(self.base_output_dir / model_name),
            "num_train_epochs": 2,  # Réduit de 3 -> 2 epochs
            "per_device_train_batch_size": batch_size["train"],
            "per_device_eval_batch_size": batch_size["eval"],
            "gradient_accumulation_steps": 2,  # Accumulation pour batch effectif plus grand
            "warmup_ratio": 0.1,  # Warmup proportionnel
            "weight_decay": 0.01,
            "learning_rate": 3e-4,  # LR légèrement plus élevé
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "logging_dir": str(self.base_output_dir / model_name / "logs"),
            "logging_steps": 25,  # Moins de logging
            "eval_steps": 100,  # Évaluation moins fréquente
            "save_steps": 200,  # Sauvegarde moins fréquente
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "save_total_limit": 2,  # Limite les checkpoints
            "dataloader_pin_memory": torch.cuda.is_available(),
            "dataloader_num_workers": 2 if torch.cuda.is_available() else 0,
            "remove_unused_columns": False,
            "report_to": [],  # Pas de logging externe
        }

        # Ajout de la précision mixte si GPU disponible
        if self.use_mixed_precision:
            base_args.update(
                {
                    "fp16": True,
                    "fp16_opt_level": "O1",
                    "dataloader_pin_memory": True,
                }
            )

        # Arguments spécifiques au type de modèle
        if model_type == "seq2seq":
            base_args.update(
                {
                    "predict_with_generate": True,
                    "generation_max_length": config["max_target_length"],
                    "metric_for_best_model": "exact_match",
                    "greater_is_better": True,
                }
            )
            return Seq2SeqTrainingArguments(**base_args)
        else:
            base_args.update(
                {
                    "metric_for_best_model": "accuracy",
                    "greater_is_better": True,
                }
            )
            return TrainingArguments(**base_args)

    def train_model_optimized(
        self, model_name: str, train_dataset: Dataset, val_dataset: Dataset
    ):
        """Entraînement optimisé avec early stopping"""
        logger.info(f"🚀 Entraînement optimisé de {model_name}...")

        config = self.model_configs[model_name]
        tokenizer = self._get_cached_tokenizer(config["model_name"])

        # Charger le modèle sur le bon device
        if config["type"] == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config["model_name"], num_labels=config["num_labels"]
            )

        model.to(self.device)

        # Tokeniser les données
        train_tokenized = self.tokenize_with_caching(train_dataset, model_name, config)
        val_tokenized = self.tokenize_with_caching(val_dataset, model_name, config)

        # Arguments d'entraînement optimisés
        training_args = self.get_optimized_training_args(model_name, config["type"])

        # Callbacks pour early stopping
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        # Créer le trainer approprié
        if config["type"] == "seq2seq":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding=True
            )
            self.current_tokenizer = tokenizer  # Pour les métriques

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics_seq2seq,
                callbacks=callbacks,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                tokenizer=tokenizer,
                compute_metrics=self.compute_metrics_classification,
                callbacks=callbacks,
            )

        # Entraînement avec mesure du temps
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Sauvegarder
        trainer.save_model()
        tokenizer.save_pretrained(str(self.base_output_dir / model_name))

        # Statistiques
        self.training_stats[model_name] = {
            "training_time": training_time,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "final_metrics": (
                trainer.state.log_history[-1] if trainer.state.log_history else {}
            ),
            "total_steps": trainer.state.global_step,
        }

        logger.info(
            f"✅ {model_name} entraîné en {training_time:.2f}s ({trainer.state.global_step} steps)"
        )
        return model, tokenizer

    def compute_metrics_seq2seq(self, eval_pred):
        """Métriques optimisées pour seq2seq"""
        predictions, labels = eval_pred

        # Limiter l'évaluation pour la vitesse
        max_eval_samples = 100
        if len(predictions) > max_eval_samples:
            predictions = predictions[:max_eval_samples]
            labels = labels[:max_eval_samples]

        decoded_preds = self.current_tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels = np.where(labels != -100, labels, self.current_tokenizer.pad_token_id)
        decoded_labels = self.current_tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Exact match simplifié
        exact_matches = sum(
            pred.strip().lower() == label.strip().lower()
            for pred, label in zip(decoded_preds, decoded_labels, strict=True)
        )
        exact_match_score = exact_matches / len(decoded_preds)

        return {"exact_match": exact_match_score}

    def compute_metrics_classification(self, eval_pred):
        """Métriques optimisées pour classification"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {"accuracy": accuracy_score(labels, predictions)}

    def train_all_models_optimized(self, dataset_dir: str = "datasets/training_fr"):
        """Version optimisée de l'entraînement de tous les modèles"""
        logger.info("🚀 Démarrage de l'entraînement optimisé...")

        # Chargement parallèle des datasets
        datasets = self.load_datasets_parallel(dataset_dir)

        if not datasets:
            logger.error("❌ Aucun dataset trouvé !")
            return {}

        trained_models = {}
        total_start_time = time.time()

        # Entraînement séquentiel optimisé (pour éviter les conflits GPU)
        for model_name, model_data in datasets.items():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🎯 Entraînement optimisé de {model_name}")
                logger.info(f"{'='*50}")

                # Préparation optimisée des données
                train_dataset = self.prepare_data_optimized(
                    model_data["train"], model_name
                )
                val_dataset = self.prepare_data_optimized(model_data["val"], model_name)

                logger.info(
                    f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)} échantillons"
                )

                # Entraînement optimisé
                model, tokenizer = self.train_model_optimized(
                    model_name, train_dataset, val_dataset
                )
                trained_models[model_name] = (model, tokenizer)

                # Nettoyage mémoire
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement de {model_name}: {e}")
                continue

        total_time = time.time() - total_start_time
        logger.info(f"\n🎉 Entraînement terminé en {total_time:.2f}s au total")

        # Rapport final optimisé
        self.print_optimized_report()
        self.save_training_config()

        return trained_models

    def print_optimized_report(self):
        """Rapport optimisé avec plus de détails sur les performances"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 RAPPORT D'ENTRAÎNEMENT OPTIMISÉ")
        logger.info("=" * 60)

        total_time = sum(
            stats["training_time"] for stats in self.training_stats.values()
        )
        total_samples = sum(
            stats["train_samples"] for stats in self.training_stats.values()
        )

        logger.info(f"⏱️  Temps total: {total_time:.2f}s")
        logger.info(f"📈 Échantillons totaux: {total_samples}")
        logger.info(
            f"🚀 Vitesse moyenne: {total_samples/total_time:.1f} échantillons/sec"
        )

        for model_name, stats in self.training_stats.items():
            logger.info(f"\n🎯 {model_name.upper()}")
            logger.info(f"   ⏱️  Temps: {stats['training_time']:.2f}s")
            logger.info(
                f"   📊 Train/Val: {stats['train_samples']}/{stats['val_samples']}"
            )
            logger.info(f"   🔄 Steps: {stats.get('total_steps', 'N/A')}")

            # Vitesse d'entraînement
            speed = stats["train_samples"] / stats["training_time"]
            logger.info(f"   🚀 Vitesse: {speed:.1f} échantillons/sec")

    def save_training_config(self):
        """Sauvegarde la configuration avec optimisations"""
        config_file = self.base_output_dir / "training_config_optimized.json"

        config_data = {
            "model_configs": self.model_configs,
            "training_stats": self.training_stats,
            "optimizations": {
                "mixed_precision": self.use_mixed_precision,
                "device": str(self.device),
                "parallel_loading": True,
                "tokenizer_caching": True,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Configuration optimisée sauvegardée: {config_file}")


def main():
    """Fonction principale optimisée"""
    logger.info("🚀 Démarrage de l'entraînement OPTIMISÉ des modèles d'IA pour Rosetta")

    # Initialiser le trainer optimisé
    trainer = OptimizedAIModelsTrainer(use_gpu=True)

    # Entraîner tous les modèles avec optimisations
    trained_models = trainer.train_all_models_optimized()

    logger.info("🎉 Entraînement optimisé terminé ! Performances améliorées.")
    return trained_models


if __name__ == "__main__":
    main()
