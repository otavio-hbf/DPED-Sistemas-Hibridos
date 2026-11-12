# main.py
# Main script for running experiments with Differentially Private Embeddings

import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
from transformers import BertForSequenceClassification

# Import our modules
from config import Config
from utils import set_seed, get_device, plot_privacy_utility_tradeoff
from data_loader import load_glue_dataset, load_conll_dataset, create_public_dataset
# Updated import from combined models file
from models import StudentModel, BertForTokenClassification, TeacherEnsemble, distill_embeddings, pate_distillation
from training import train_model_with_dpsgd, train_model_without_dp, test_model

def parse_args():
    parser = argparse.ArgumentParser(description="Differentially Private Embeddings for NLP")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="glue", choices=["glue", "conll2003"],
                        help="Dataset to use (glue or conll2003)")
    parser.add_argument("--task", type=str, default="sst2", 
                        choices=["sst2", "qqp", "mnli", "cola"],
                        help="Task name for GLUE benchmark")
    
    # Model parameters
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="BERT model to use")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Privacy parameters
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="Privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Privacy parameter delta")
    parser.add_argument("--no_dp", action="store_true",
                        help="Disable differential privacy (for baseline)")
    
    # Teacher-Student parameters
    parser.add_argument("--num_teachers", type=int, default=5,
                        help="Number of teacher models")
    parser.add_argument("--teacher_noise", type=float, default=0.5,
                        help="Noise for teacher output aggregation")
    parser.add_argument("--rare_token_threshold", type=int, default=2,
                        help="Threshold for rare token detection")
    parser.add_argument("--rare_token_noise_factor", type=float, default=2.0,
                        help="Noise factor for rare tokens")
    
    # Multi-layer DP parameters
    parser.add_argument("--no_multi_layer_noise", action="store_true",
                        help="Disable multi-layer noise injection")
    parser.add_argument("--embedding_noise_std", type=float, default=0.1,
                        help="Standard deviation of noise at embedding layer")
    parser.add_argument("--intermediate_noise_std", type=float, default=0.05,
                        help="Standard deviation of noise at intermediate layer")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Fast-check dataset subset parameters
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples (None uses full split)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of evaluation samples (None uses full split)")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Maximum number of test samples (None uses full split)")
    parser.add_argument("--max_public_samples", type=int, default=10000,
                        help="Maximum number of public samples for distillation")
    parser.add_argument("--subset_strategy", type=str, default="random", choices=["random", "head"],
                        help="Subset strategy for max_*_samples")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Seed used when subset_strategy is random")
    
    # Experiment parameters
    parser.add_argument("--method", type=str, default="dp-distill",
                        choices=["non-private", "dp-sgd", "pate-distill", "dp-distill"],
                        help="Method to use")
    parser.add_argument("--run_all_methods", action="store_true",
                        help="Run all methods for comparison")
    parser.add_argument("--privacy_utility_tradeoff", action="store_true",
                        help="Run privacy-utility tradeoff experiment")
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update config with command line arguments"""
    
    # Dataset parameters
    config.dataset_name = args.dataset
    config.task_name = args.task
    
    # Model parameters
    config.bert_model = args.bert_model
    config.max_seq_length = args.max_seq_length
    
    # Privacy parameters
    config.epsilon = args.epsilon
    config.delta = args.delta
    config.enable_dp = not args.no_dp
    
    # Teacher-Student parameters
    config.num_teachers = args.num_teachers
    config.teacher_aggregation_noise = args.teacher_noise
    config.rare_token_threshold = args.rare_token_threshold
    config.rare_token_noise_factor = args.rare_token_noise_factor
    
    # Multi-layer DP parameters
    config.multi_layer_noise = not args.no_multi_layer_noise
    config.embedding_noise_std = args.embedding_noise_std
    config.intermediate_noise_std = args.intermediate_noise_std
    
    # Training parameters
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.weight_decay = args.weight_decay

    # Subset parameters
    config.max_train_samples = args.max_train_samples
    config.max_eval_samples = args.max_eval_samples
    config.max_test_samples = args.max_test_samples
    config.max_public_samples = args.max_public_samples
    config.random_subset = args.subset_strategy == "random"
    config.subset_seed = args.subset_seed
    
    return config


def build_dp_compatible_model(model, config, device):
    """Convert custom student to HF classifier for Opacus compatibility."""
    if not isinstance(model, StudentModel):
        return model

    num_labels = 2 if config.task_name in ["sst2", "qqp", "cola"] else 3
    dp_model = BertForSequenceClassification.from_pretrained(
        config.bert_model,
        num_labels=num_labels
    ).to(device)

    src_state = model.state_dict()
    dst_state = dp_model.state_dict()
    for name in dst_state.keys():
        if name in src_state and src_state[name].shape == dst_state[name].shape:
            dst_state[name] = src_state[name].detach().clone()
    dp_model.load_state_dict(dst_state)

    return dp_model


def save_results_summary(args, config, experiment_records):
    """Persist a full experiment summary with parameters and key metrics."""
    summary_dir = os.path.join("results", config.dataset_name)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "results_summary.txt")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Execution timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Dataset: {config.dataset_name}\n")
        f.write(f"Task: {config.task_name}\n")
        f.write("\nParameters used:\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"- {key}: {value}\n")

        f.write("\nMetrics by run:\n")
        for i, record in enumerate(experiment_records, start=1):
            metrics = record["metrics"]
            accuracy = metrics.get("accuracy", float("nan"))
            precision = metrics.get("precision", float("nan"))
            recall = metrics.get("recall", float("nan"))
            f1 = metrics.get("f1", float("nan"))

            f.write(
                f"Run {i}: method={record['method']}, epsilon={record['epsilon']}, "
                f"enable_dp={record['enable_dp']}\n"
            )
            f.write(f"  Accuracy: {accuracy:.6f}\n")
            f.write(f"  Precision: {precision:.6f}\n")
            f.write(f"  Recall: {recall:.6f}\n")
            f.write(f"  F1: {f1:.6f}\n")

        f.write("=" * 100 + "\n")

    print(f"Saved automatic summary to: {summary_path}")


def is_dp_effectively_enabled(method, config):
    """Return whether a method effectively runs DP during fine-tuning."""
    if method == "non-private":
        return False
    if method == "dp-sgd":
        return True
    return config.enable_dp


def _unwrap_model(model):
    """Unwrap wrappers (e.g., Opacus/DP wrappers) to access the underlying model."""
    current = model
    while hasattr(current, "_module"):
        current = current._module
    return current


def _extract_word_embeddings(model):
    """Extract the token embedding matrix from BERT-based models."""
    base_model = _unwrap_model(model)
    if hasattr(base_model, "bert") and hasattr(base_model.bert, "embeddings"):
        return base_model.bert.embeddings.word_embeddings.weight.detach().cpu()
    raise ValueError("Could not find BERT word embeddings in model.")


def _extract_split_labels(dataset):
    """Extract labels from a split as a numpy array and return (label_key, labels)."""
    if dataset is None or len(dataset) == 0:
        return None, None

    sample = dataset[0]
    label_key = "label" if "label" in sample else ("labels" if "labels" in sample else None)
    if label_key is None:
        return None, None

    labels = []
    for i in range(len(dataset)):
        value = dataset[i][label_key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        labels.append(value)

    return label_key, np.asarray(labels)


def _extract_label_mapping(dataset, label_key):
    """Return {id: name} mapping when dataset has ClassLabel metadata."""
    if dataset is None or label_key is None:
        return None
    if not hasattr(dataset, "features") or label_key not in dataset.features:
        return None

    feature = dataset.features[label_key]
    names = getattr(feature, "names", None)
    if not names:
        return None

    return {str(i): name for i, name in enumerate(names)}


def save_model_embeddings(model, config, method, train_dataset=None, eval_dataset=None, test_dataset=None):
    """Save final token embeddings for a run in a folder named by dataset/method/epsilon."""
    epsilon_tag = str(config.epsilon).replace(".", "p")
    run_dir_name = f"{config.dataset_name}_{config.task_name}_{method}_eps{epsilon_tag}"
    output_dir = os.path.join("results", "embeddings", run_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    embeddings = _extract_word_embeddings(model)
    embeddings_pt_path = os.path.join(output_dir, "token_embeddings.pt")
    embeddings_npy_path = os.path.join(output_dir, "token_embeddings.npy")
    metadata_path = os.path.join(output_dir, "metadata.txt")
    label_map_path = os.path.join(output_dir, "label_id_to_name.json")

    torch.save(embeddings, embeddings_pt_path)
    np.save(embeddings_npy_path, embeddings.numpy())

    split_info = {}
    for split_name, split_dataset in [
        ("train", train_dataset),
        ("eval", eval_dataset),
        ("test", test_dataset),
    ]:
        label_key, labels_array = _extract_split_labels(split_dataset)
        if labels_array is None:
            continue

        labels_path = os.path.join(output_dir, f"{split_name}_labels.npy")
        np.save(labels_path, labels_array)
        split_info[split_name] = {
            "label_key": label_key,
            "shape": tuple(labels_array.shape),
            "file": os.path.basename(labels_path),
        }

    label_mapping = _extract_label_mapping(test_dataset, "label")
    if label_mapping is None:
        label_mapping = _extract_label_mapping(test_dataset, "labels")
    if label_mapping is not None:
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, ensure_ascii=True, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"saved_at={datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"dataset={config.dataset_name}\n")
        f.write(f"task={config.task_name}\n")
        f.write(f"method={method}\n")
        f.write(f"epsilon={config.epsilon}\n")
        f.write(f"shape={tuple(embeddings.shape)}\n")
        f.write("files=token_embeddings.pt,token_embeddings.npy\n")
        if split_info:
            for split_name, info in split_info.items():
                f.write(
                    f"labels_{split_name}=key:{info['label_key']},shape:{info['shape']},file:{info['file']}\n"
                )
        if label_mapping is not None:
            f.write(f"label_id_to_name_file={os.path.basename(label_map_path)}\n")

    print(f"Saved model embeddings to: {output_dir}")


def run_method(method, train_dataset, eval_dataset, test_dataset, public_dataset, config, device):
    """
    Run a specific method for training with differential privacy.
    
    Args:
        method: Method name (non-private, dp-sgd, pate-distill, dp-distill)
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset
        public_dataset: Public/unlabeled dataset for distillation
        config: Configuration object
        device: Device to run on
        
    Returns:
        Trained model and test metrics
    """
    print(f"\n{'='*80}")
    print(f"Running method: {method}")
    print(f"Task: {config.task_name}")
    print(f"Epsilon: {config.epsilon}")
    print(f"{'='*80}")
    
    # Create appropriate model based on dataset and task
    if config.dataset_name == "conll2003":
        model = BertForTokenClassification(config).to(device)
    else:
        if method == "dp-distill" or method == "pate-distill":
            model = StudentModel(config).to(device)
        else:
            model = BertForSequenceClassification.from_pretrained(
                config.bert_model,
                num_labels=2 if config.task_name in ["sst2", "qqp", "cola"] else 3
            ).to(device)
    
    # Train based on method
    if method == "non-private":
        # Standard training without DP
        config_copy = Config()
        for key, value in vars(config).items():
            setattr(config_copy, key, value)
        config_copy.enable_dp = False
        
        model, _ = train_model_without_dp(model, train_dataset, eval_dataset, config_copy, device)
    
    elif method == "dp-sgd":
        # DP-SGD training
        config_copy = Config()
        for key, value in vars(config).items():
            setattr(config_copy, key, value)
        config_copy.enable_dp = True
        
        model, _ = train_model_with_dpsgd(model, train_dataset, eval_dataset, config_copy, device)
    
    elif method == "pate-distill":
        # PATE-based distillation
        config_copy = Config()
        for key, value in vars(config).items():
            setattr(config_copy, key, value)
        config_copy.enable_dp = config.enable_dp
        config_copy.multi_layer_noise = False  # No multi-layer noise for PATE baseline
        
        # Create and train teacher ensemble
        teacher_ensemble = TeacherEnsemble(config_copy, train_dataset, device)
        teacher_ensemble.train_teachers()
        
        # Train student with PATE-style distillation
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=config_copy.batch_size, shuffle=True)
        model = pate_distillation(teacher_ensemble, model, public_dataset, train_loader, config_copy, device)

        # Switch to Opacus-friendly classifier before DP-SGD.
        model = build_dp_compatible_model(model, config_copy, device)
        
        # Fine-tune with DP-SGD only when DP is enabled.
        if config_copy.enable_dp:
            model, _ = train_model_with_dpsgd(model, train_dataset, eval_dataset, config_copy, device)
        else:
            model, _ = train_model_without_dp(model, train_dataset, eval_dataset, config_copy, device)
    
    elif method == "dp-distill":
        # Our proposed method: Teacher-Student Distillation with Multi-Layer DP
        # Create and train teacher ensemble
        teacher_ensemble = TeacherEnsemble(config, train_dataset, device)
        teacher_ensemble.train_teachers()
        
        # First distill embeddings
        model = distill_embeddings(teacher_ensemble, model, public_dataset, config, device)

        # Switch to Opacus-friendly classifier before DP-SGD.
        model = build_dp_compatible_model(model, config, device)
        
        # Then fine-tune with DP-SGD (or standard training when --no_dp is set).
        if config.enable_dp:
            model, _ = train_model_with_dpsgd(model, train_dataset, eval_dataset, config, device)
        else:
            model, _ = train_model_without_dp(model, train_dataset, eval_dataset, config, device)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Test the model
    test_metrics = test_model(model, test_dataset, config, device)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"Final Results Summary for {method}:")
    print(f"Task: {config.task_name}")
    print(f"Epsilon: {config.epsilon}")
    print("-"*80)
    for metric_name, value in test_metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")
    print(f"{'='*80}\n")

    # Save final model embeddings for this method run.
    save_model_embeddings(
        model,
        config,
        method,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
    )
    
    return model, test_metrics


def run_privacy_utility_tradeoff(train_dataset, eval_dataset, test_dataset, public_dataset, config, device):
    """
    Run privacy-utility tradeoff experiment with different epsilon values.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset
        public_dataset: Public/unlabeled dataset for distillation
        config: Configuration object
        device: Device to run on
    """
    print("\nRunning privacy-utility tradeoff experiment...")
    
    # Define methods to compare
    methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
    
    # Define epsilon values to test
    epsilon_values = [1.0, 2.0, 4.0, 8.0, 16.0]
    
    # Store metrics for each method and epsilon
    results = {method: [] for method in methods}
    experiment_records = []
    
    # Run each method with each epsilon value
    for epsilon in epsilon_values:
        print(f"\n{'='*80}")
        print(f"Running with epsilon = {epsilon}")
        print(f"{'='*80}")
        
        # Update config with current epsilon
        config.epsilon = epsilon
        config.update_epsilon(epsilon)
        
        # Run each method
        for method in methods:
            if method == "non-private":
                # Non-private method only needs to be run once
                if not results[method]:
                    _, metrics = run_method(method, train_dataset, eval_dataset, test_dataset, 
                                          public_dataset, config, device)
                    experiment_records.append({
                        "method": method,
                        "epsilon": epsilon,
                        "enable_dp": is_dp_effectively_enabled(method, config),
                        "metrics": metrics,
                    })
                    # Repeat the same metrics for all epsilon values
                    results[method] = [metrics] * len(epsilon_values)
                continue
            
            # Run method with current epsilon
            _, metrics = run_method(method, train_dataset, eval_dataset, test_dataset, 
                                  public_dataset, config, device)
            results[method].append(metrics)
            experiment_records.append({
                "method": method,
                "epsilon": epsilon,
                "enable_dp": is_dp_effectively_enabled(method, config),
                "metrics": metrics,
            })
    
    # Extract primary metric values (accuracy or F1)
    primary_metric = "accuracy" if config.task_name in ["sst2", "mnli", "cola"] else "f1"
    metric_values = {
        method: [metrics[primary_metric] for metrics in method_results]
        for method, method_results in results.items()
    }
    
    # Plot results
    plot_privacy_utility_tradeoff(epsilon_values, metric_values, methods, primary_metric)
    
    # Save results to file
    with open(f"privacy_utility_tradeoff_{config.task_name}.txt", "w") as f:
        f.write(f"Privacy-Utility Tradeoff Experiment Results for {config.task_name}\n")
        f.write(f"Primary metric: {primary_metric}\n\n")
        
        f.write("Epsilon values: " + ", ".join(str(eps) for eps in epsilon_values) + "\n\n")
        
        for method in methods:
            f.write(f"{method}: " + ", ".join(f"{value:.4f}" for value in metric_values[method]) + "\n")

    return experiment_records


def main():
    """Main function to run experiments"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create config
    config = Config()
    config = update_config_from_args(config, args)
    
    # Load datasets
    if config.dataset_name == "glue":
        train_dataset, eval_dataset, test_dataset = load_glue_dataset(config)
    elif config.dataset_name == "conll2003":
        train_dataset, eval_dataset, test_dataset = load_conll_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create public dataset for distillation
    public_dataset = create_public_dataset(config, size=config.max_public_samples)
    print(f"Public dataset size: {len(public_dataset)}")
    
    # Run experiments
    experiment_records = []
    if args.privacy_utility_tradeoff:
        # Run privacy-utility tradeoff experiment
        experiment_records = run_privacy_utility_tradeoff(
            train_dataset, eval_dataset, test_dataset, public_dataset, config, device
        )
    elif args.run_all_methods:
        # Run all methods for comparison
        methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
        results = {}
        
        for method in methods:
            _, metrics = run_method(
                method, train_dataset, eval_dataset, test_dataset, public_dataset, config, device
            )
            results[method] = metrics
            experiment_records.append({
                "method": method,
                "epsilon": config.epsilon,
                "enable_dp": is_dp_effectively_enabled(method, config),
                "metrics": metrics,
            })
        
        # Print comparison
        print("\nResults Summary:")
        print("-" * 80)
        for method, metrics in results.items():
            print(f"{method}: {metrics}")
    else:
        # Run single method
        _, metrics = run_method(
            args.method, train_dataset, eval_dataset, test_dataset, public_dataset, config, device
        )
        experiment_records.append({
            "method": args.method,
            "epsilon": config.epsilon,
            "enable_dp": is_dp_effectively_enabled(args.method, config),
            "metrics": metrics,
        })

    # Always save full summary at the end of execution.
    save_results_summary(args, config, experiment_records)


if __name__ == "__main__":
    main()