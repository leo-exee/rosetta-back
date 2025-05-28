import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    pipeline,
)

from datasets import Dataset, DatasetDict

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModelsTrainer:
    """
    Entraîneur pour les trois modèles d'IA :
    1. Fill-in-the-Blank Generator (T5)
    2. Sentence Scrambler (T5)
    3. Definition Matcher (BERT)
    """

    def __init__(self, base_output_dir: str = "models/trained"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration des modèles
        self.model_configs = {
            "fill_in_blank": {
                "model_name": "google/flan-t5-small",
                "type": "seq2seq",
                "task_prefix": "Fill in the blanks: ",
                "max_input_length": 256,
                "max_target_length": 128,
            },
            "sentence_scrambler": {
                "model_name": "google/flan-t5-small",
                "type": "seq2seq",
                "task_prefix": "Unscramble sentence: ",
                "max_input_length": 128,
                "max_target_length": 128,
            },
            "definition_matcher": {
                "model_name": "distilbert-base-uncased",
                "type": "classification",
                "num_labels": 3,  # 3 mots à matcher
                "max_input_length": 256,
            },
        }

        # Statistiques d'entraînement
        self.training_stats = {}

    def load_datasets(
        self, dataset_dir: str = "datasets/training_fr"
    ) -> dict[str, DatasetDict]:
        """Charge tous les datasets d'entraînement"""
        logger.info("📚 Chargement des datasets...")

        datasets = {}
        dataset_path = Path(dataset_dir)

        for model_name in self.model_configs.keys():
            model_dir = dataset_path / model_name

            if not model_dir.exists():
                logger.warning(f"⚠️ Dossier non trouvé: {model_dir}")
                continue

            # Charger train/val/test
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

            if splits:
                datasets[model_name] = splits

        return datasets

    def prepare_fill_in_blank_data(self, data: list[dict]) -> Dataset:
        """Prépare les données pour le modèle Fill-in-the-Blank"""
        inputs = []
        targets = []

        for item in data:
            try:
                # Format d'entrée: context|level|original_text
                input_parts = item["input"].split("|")
                if len(input_parts) != 3:
                    logger.warning(f"Format d'entrée invalide: {item['input']}")
                    continue

                context, level, original_text = input_parts

                # Format de sortie: text_with_blanks|||words_to_fill|||complete_text
                output_parts = item["output"].split("|||")
                if len(output_parts) != 3:
                    logger.warning(f"Format de sortie invalide: {item['output']}")
                    continue

                text_with_blanks, words_to_fill, complete_text = output_parts

                # Input pour le modèle: instruction + contexte + texte à trous
                model_input = f"Context: {context}, Level: {level}. Fill the blanks: {text_with_blanks}. Options: {words_to_fill}"

                # Target: texte complet
                model_target = complete_text.strip()

                # Validation des données
                if len(model_input.strip()) == 0 or len(model_target.strip()) == 0:
                    logger.warning("Données vides détectées, ignorées")
                    continue

                inputs.append(model_input.strip())
                targets.append(model_target.strip())

            except Exception as e:
                logger.warning(f"Erreur lors du traitement d'un item: {e}")
                continue

        logger.info(f"📊 Fill-in-blank: {len(inputs)} exemples préparés")
        return Dataset.from_dict({"input_text": inputs, "target_text": targets})

    def prepare_sentence_scrambler_data(self, data: list[dict]) -> Dataset:
        """Prépare les données pour le modèle Sentence Scrambler"""
        inputs = []
        targets = []

        for item in data:
            try:
                # Format d'entrée: context|level|original_sentence
                input_parts = item["input"].split("|")
                if len(input_parts) != 3:
                    logger.warning(f"Format d'entrée invalide: {item['input']}")
                    continue

                context, level, original_sentence = input_parts

                # Format de sortie: scrambled_words|||original_sentence
                output_parts = item["output"].split("|||")
                if len(output_parts) != 2:
                    logger.warning(f"Format de sortie invalide: {item['output']}")
                    continue

                scrambled_words, target_sentence = output_parts

                # Input pour le modèle: instruction + mots mélangés
                model_input = (
                    f"Context: {context}, Level: {level}. Unscramble: {scrambled_words}"
                )

                # Target: phrase correcte
                model_target = target_sentence.strip()

                # Validation des données
                if len(model_input.strip()) == 0 or len(model_target.strip()) == 0:
                    logger.warning("Données vides détectées, ignorées")
                    continue

                inputs.append(model_input.strip())
                targets.append(model_target.strip())

            except Exception as e:
                logger.warning(f"Erreur lors du traitement d'un item: {e}")
                continue

        logger.info(f"📊 Sentence scrambler: {len(inputs)} exemples préparés")
        return Dataset.from_dict({"input_text": inputs, "target_text": targets})

    def prepare_definition_matcher_data(self, data: list[dict]) -> Dataset:
        """Prépare les données pour le modèle Definition Matcher"""
        inputs = []
        labels = []

        for item in data:
            try:
                # Format d'entrée: context|level
                input_parts = item["input"].split("|")
                if len(input_parts) != 2:
                    logger.warning(f"Format d'entrée invalide: {item['input']}")
                    continue

                context, level = input_parts

                # Format de sortie: word1,word2,word3|||def1|||def2|||def3|||1,2,3
                output_parts = item["output"].split("|||")
                if len(output_parts) != 5:
                    logger.warning(f"Format de sortie invalide: {item['output']}")
                    continue

                words_str, def1, def2, def3, correct_matches = output_parts
                words = [w.strip() for w in words_str.split(",")]
                definitions = [def1.strip(), def2.strip(), def3.strip()]

                try:
                    matches = [
                        int(x.strip()) - 1 for x in correct_matches.split(",")
                    ]  # Convert to 0-based
                except ValueError as e:
                    logger.warning(f"Erreur de conversion des correspondances: {e}")
                    continue

                # Créer des exemples pour chaque association mot-définition
                for _i, (word, correct_def_idx) in enumerate(
                    zip(words, matches, strict=False)
                ):
                    if correct_def_idx < 0 or correct_def_idx >= len(definitions):
                        logger.warning(
                            f"Index de définition invalide: {correct_def_idx}"
                        )
                        continue

                    # Input: contexte + mot + toutes les définitions
                    all_defs = " | ".join(definitions)
                    model_input = f"Context: {context}, Level: {level}. Word: {word}. Definitions: {all_defs}"

                    # Label: index de la bonne définition
                    label = correct_def_idx

                    # Validation des données
                    if len(model_input.strip()) == 0:
                        logger.warning("Données d'entrée vides détectées, ignorées")
                        continue

                    inputs.append(model_input.strip())
                    labels.append(label)

            except Exception as e:
                logger.warning(f"Erreur lors du traitement d'un item: {e}")
                continue

        logger.info(f"📊 Definition matcher: {len(inputs)} exemples préparés")
        return Dataset.from_dict({"input_text": inputs, "labels": labels})

    def tokenize_seq2seq_data(
        self, dataset: Dataset, tokenizer, config: dict
    ) -> Dataset:
        """Tokenise les données pour les modèles seq2seq"""

        def tokenize_function(examples):
            # Ajouter le préfixe de tâche aux inputs
            prefixed_inputs = [
                config["task_prefix"] + text for text in examples["input_text"]
            ]

            # Tokeniser les inputs
            model_inputs = tokenizer(
                prefixed_inputs,
                max_length=config["max_input_length"],
                truncation=True,
                padding=True,  # Changé de "max_length" à True pour un padding dynamique
                return_tensors=None,  # Ajouté pour éviter les problèmes de format
            )

            # Tokeniser les targets
            with tokenizer.as_target_tokenizer():
                targets = tokenizer(
                    examples["target_text"],
                    max_length=config["max_target_length"],
                    truncation=True,
                    padding=True,  # Changé de "max_length" à True
                    return_tensors=None,  # Ajouté pour éviter les problèmes de format
                )

            # Remplacer les pad tokens par -100 dans les labels pour ignorer dans la loss
            labels = []
            for target_ids in targets["input_ids"]:
                label_ids = [
                    (token_id if token_id != tokenizer.pad_token_id else -100)
                    for token_id in target_ids
                ]
                labels.append(label_ids)

            model_inputs["labels"] = labels
            return model_inputs

        # Appliquer la tokenisation par batch
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,  # Supprimer les colonnes originales
            desc="Tokenizing dataset",
        )

        return tokenized_dataset

    def tokenize_classification_data(
        self, dataset: Dataset, tokenizer, config: dict
    ) -> Dataset:
        """Tokenise les données pour le modèle de classification"""

        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["input_text"],
                max_length=config["max_input_length"],
                truncation=True,
                padding=True,  # Changé de "max_length" à True
                return_tensors=None,  # Ajouté pour éviter les problèmes de format
            )

            # Garder les labels
            tokenized["labels"] = examples["labels"]
            return tokenized

        # Appliquer la tokenisation par batch
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["input_text"],  # Supprimer seulement la colonne input_text
            desc="Tokenizing dataset",
        )

        return tokenized_dataset

    def compute_metrics_seq2seq(self, eval_pred):
        """Calcule les métriques pour les modèles seq2seq"""
        predictions, labels = eval_pred

        # CORRECTION: Vérifier et nettoyer les prédictions
        # Les prédictions peuvent contenir des valeurs hors plage
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Convertir en numpy array si nécessaire
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Nettoyer les prédictions : remplacer les valeurs invalides
        # Les token IDs valides sont généralement entre 0 et vocab_size
        vocab_size = self.current_tokenizer.vocab_size

        # Clipper les prédictions pour qu'elles soient dans la plage valide
        predictions = np.clip(predictions, 0, vocab_size - 1)

        # Prendre l'argmax si les prédictions sont des logits
        if len(predictions.shape) > 2:
            predictions = np.argmax(predictions, axis=-1)

        try:
            # Décoder les prédictions avec gestion d'erreur
            decoded_preds = []
            for pred in predictions:
                try:
                    # Filtrer les tokens invalides avant le décodage
                    valid_pred = pred[pred >= 0]  # Enlever les tokens négatifs
                    valid_pred = valid_pred[
                        valid_pred < vocab_size
                    ]  # Enlever les tokens trop grands
                    decoded_pred = self.current_tokenizer.decode(
                        valid_pred, skip_special_tokens=True
                    )
                    decoded_preds.append(decoded_pred)
                except Exception:
                    # En cas d'erreur, utiliser une chaîne vide
                    decoded_preds.append("")

            # Traiter les labels
            labels = np.where(
                labels != -100, labels, self.current_tokenizer.pad_token_id
            )

            decoded_labels = []
            for label in labels:
                try:
                    # Même traitement pour les labels
                    valid_label = label[label >= 0]
                    valid_label = valid_label[valid_label < vocab_size]
                    decoded_label = self.current_tokenizer.decode(
                        valid_label, skip_special_tokens=True
                    )
                    decoded_labels.append(decoded_label)
                except Exception:
                    decoded_labels.append("")

        except Exception as e:
            # En cas d'erreur complète, retourner des métriques par défaut
            return {"exact_match": 0.0, "bleu_approx": 0.0, "decode_error": str(e)}

        # Calculer l'exactitude exacte (exact match)
        exact_matches = sum(
            pred.strip() == label.strip()
            for pred, label in zip(decoded_preds, decoded_labels, strict=True)
            if pred and label  # Éviter les chaînes vides
        )
        exact_match_score = exact_matches / len(decoded_preds) if decoded_preds else 0.0

        # Calculer BLEU approximatif (simple overlap)
        bleu_scores = []
        for pred, label in zip(decoded_preds, decoded_labels, strict=True):
            if pred and label:  # Éviter les chaînes vides
                pred_words = set(pred.lower().split())
                label_words = set(label.lower().split())
                if len(label_words) > 0:
                    overlap = len(pred_words & label_words) / len(label_words)
                    bleu_scores.append(overlap)

        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

        return {"exact_match": exact_match_score, "bleu_approx": avg_bleu}

    def compute_metrics_classification(self, eval_pred):
        """Calcule les métriques pour le modèle de classification"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    def train_seq2seq_model(
        self, model_name: str, train_dataset: Dataset, val_dataset: Dataset
    ) -> tuple[object, object]:
        """Entraîne un modèle seq2seq (T5)"""
        logger.info(f"🚀 Entraînement du modèle {model_name}...")

        config = self.model_configs[model_name]

        # Charger le modèle et tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

        # Ajouter un token de padding si nécessaire
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))

        # Stocker le tokenizer pour les métriques
        self.current_tokenizer = tokenizer

        # Tokeniser les données
        logger.info("🔤 Tokenisation des données...")
        train_tokenized = self.tokenize_seq2seq_data(train_dataset, tokenizer, config)
        val_tokenized = self.tokenize_seq2seq_data(val_dataset, tokenizer, config)

        # Configuration d'entraînement
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.base_output_dir / model_name),
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Réduit encore plus pour éviter les problèmes de mémoire
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,  # Ajouté pour compenser la petite batch size
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(self.base_output_dir / model_name / "logs"),
            logging_steps=50,
            eval_steps=200,
            save_steps=400,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=config["max_target_length"],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,  # Désactiver fp16 pour éviter les problèmes
            dataloader_num_workers=0,  # Ajouté pour éviter les problèmes de multiprocessing
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,  # Ajouté pour l'efficacité
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics_seq2seq,
        )

        # Entraînement
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Sauvegarder le modèle
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
        }

        logger.info(f"✅ {model_name} entraîné en {training_time:.2f}s")
        return model, tokenizer

    def train_classification_model(
        self, model_name: str, train_dataset: Dataset, val_dataset: Dataset
    ) -> tuple[object, object]:
        """Entraîne un modèle de classification (BERT)"""
        logger.info(f"🚀 Entraînement du modèle {model_name}...")

        config = self.model_configs[model_name]

        # Charger le modèle et tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=config["num_labels"]
        )

        # Tokeniser les données
        logger.info("🔤 Tokenisation des données...")
        train_tokenized = self.tokenize_classification_data(
            train_dataset, tokenizer, config
        )
        val_tokenized = self.tokenize_classification_data(
            val_dataset, tokenizer, config
        )

        # Configuration d'entraînement
        training_args = TrainingArguments(
            output_dir=str(self.base_output_dir / model_name),
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Réduit pour éviter les problèmes de mémoire
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,  # Ajouté pour compenser la petite batch size
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(self.base_output_dir / model_name / "logs"),
            logging_steps=50,
            eval_steps=200,
            save_steps=400,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            dataloader_pin_memory=False,
            fp16=False,  # Désactiver fp16 pour éviter les problèmes
            dataloader_num_workers=0,  # Ajouté pour éviter les problèmes de multiprocessing
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics_classification,
        )

        # Entraînement
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Sauvegarder le modèle
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
        }

        logger.info(f"✅ {model_name} entraîné en {training_time:.2f}s")
        return model, tokenizer

    def evaluate_model(self, model_name: str, test_data: list[dict]) -> dict:
        """Évalue un modèle entraîné sur les données de test"""
        logger.info(f"🧪 Évaluation du modèle {model_name}...")

        model_path = self.base_output_dir / model_name
        if not model_path.exists():
            logger.error(f"❌ Modèle non trouvé: {model_path}")
            return {}

        config = self.model_configs[model_name]

        try:
            if config["type"] == "seq2seq":
                # Modèle seq2seq
                pipe = pipeline(
                    "text2text-generation",
                    model=str(model_path),
                    tokenizer=str(model_path),
                    max_length=config["max_target_length"],
                    device=-1,  # Force CPU pour éviter les problèmes de GPU
                )

                # Préparer les données de test
                if model_name == "fill_in_blank":
                    test_dataset = self.prepare_fill_in_blank_data(test_data)
                else:  # sentence_scrambler
                    test_dataset = self.prepare_sentence_scrambler_data(test_data)

                # Évaluer
                correct = 0
                total = 0

                for i in range(min(50, len(test_dataset))):  # Limiter pour l'évaluation
                    input_text = config["task_prefix"] + test_dataset[i]["input_text"]
                    target = test_dataset[i]["target_text"]

                    prediction = pipe(input_text)[0]["generated_text"]

                    if prediction.strip().lower() == target.strip().lower():
                        correct += 1
                    total += 1

                accuracy = correct / total if total > 0 else 0
                return {"accuracy": accuracy, "total_tested": total}

            else:  # classification
                # Modèle de classification
                pipe = pipeline(
                    "text-classification",
                    model=str(model_path),
                    tokenizer=str(model_path),
                    device=-1,  # Force CPU pour éviter les problèmes de GPU
                )

                test_dataset = self.prepare_definition_matcher_data(test_data)

                correct = 0
                total = 0

                for i in range(min(50, len(test_dataset))):
                    input_text = test_dataset[i]["input_text"]
                    true_label = test_dataset[i]["labels"]

                    prediction = pipe(input_text)
                    predicted_label = int(prediction[0]["label"].split("_")[-1])

                    if predicted_label == true_label:
                        correct += 1
                    total += 1

                accuracy = correct / total if total > 0 else 0
                return {"accuracy": accuracy, "total_tested": total}

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation de {model_name}: {e}")
            return {"error": str(e)}

    def train_all_models(self, dataset_dir: str = "datasets/training_fr"):
        """Entraîne tous les modèles"""
        logger.info("🚀 Démarrage de l'entraînement de tous les modèles...")

        # Charger les datasets
        datasets = self.load_datasets(dataset_dir)

        if not datasets:
            logger.error("❌ Aucun dataset trouvé !")
            return

        trained_models = {}

        for model_name, model_data in datasets.items():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🎯 Entraînement de {model_name}")
                logger.info(f"{'='*50}")

                # Préparer les datasets
                if model_name == "fill_in_blank":
                    train_dataset = self.prepare_fill_in_blank_data(model_data["train"])
                    val_dataset = self.prepare_fill_in_blank_data(model_data["val"])
                elif model_name == "sentence_scrambler":
                    train_dataset = self.prepare_sentence_scrambler_data(
                        model_data["train"]
                    )
                    val_dataset = self.prepare_sentence_scrambler_data(
                        model_data["val"]
                    )
                elif model_name == "definition_matcher":
                    train_dataset = self.prepare_definition_matcher_data(
                        model_data["train"]
                    )
                    val_dataset = self.prepare_definition_matcher_data(
                        model_data["val"]
                    )
                else:
                    logger.warning(f"⚠️ Modèle non reconnu: {model_name}")
                    continue

                # Vérifier que les datasets ne sont pas vides
                if len(train_dataset) == 0 or len(val_dataset) == 0:
                    logger.warning(f"⚠️ Dataset vide pour {model_name}, ignoré")
                    continue

                # Entraîner le modèle
                config = self.model_configs[model_name]
                if config["type"] == "seq2seq":
                    model, tokenizer = self.train_seq2seq_model(
                        model_name, train_dataset, val_dataset
                    )
                else:  # classification
                    model, tokenizer = self.train_classification_model(
                        model_name, train_dataset, val_dataset
                    )

                trained_models[model_name] = (model, tokenizer)

                # Évaluer sur les données de test si disponibles
                if "test" in model_data:
                    eval_results = self.evaluate_model(model_name, model_data["test"])
                    self.training_stats[model_name]["test_results"] = eval_results
                    logger.info(
                        f"📊 Résultats de test pour {model_name}: {eval_results}"
                    )

            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement de {model_name}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                continue

        # Rapport final
        self.print_training_report()

        return trained_models

    def print_training_report(self):
        """Affiche un rapport complet de l'entraînement"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 RAPPORT D'ENTRAÎNEMENT FINAL")
        logger.info("=" * 60)

        for model_name, stats in self.training_stats.items():
            logger.info(f"\n🎯 {model_name.upper()}")
            logger.info(f"   Temps d'entraînement: {stats['training_time']:.2f}s")
            logger.info(f"   Échantillons train: {stats['train_samples']}")
            logger.info(f"   Échantillons val: {stats['val_samples']}")

            if "test_results" in stats:
                test_acc = stats["test_results"].get("accuracy", 0)
                logger.info(f"   Précision sur test: {test_acc:.2%}")

        logger.info(
            "\n✅ Tous les modèles sont sauvegardés dans: " + str(self.base_output_dir)
        )

    def save_training_config(self):
        """Sauvegarde la configuration d'entraînement"""
        config_file = self.base_output_dir / "training_config.json"

        config_data = {
            "model_configs": self.model_configs,
            "training_stats": self.training_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Configuration sauvegardée: {config_file}")


def main():
    """Fonction principale pour lancer l'entraînement"""
    logger.info("🚀 Démarrage de l'entraînement des modèles d'IA pour Rosetta")

    # Vérifier la disponibilité de GPU
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU détecté: {torch.cuda.get_device_name()}")
    else:
        logger.info("💻 Entraînement sur CPU")

    # Initialiser le trainer
    trainer = AIModelsTrainer()

    # Entraîner tous les modèles
    trained_models = trainer.train_all_models()

    # Sauvegarder la configuration
    trainer.save_training_config()

    logger.info(
        "🎉 Entraînement français terminé ! Les modèles sont prêts à générer des exercices."
    )

    return trained_models


if __name__ == "__main__":
    main()
