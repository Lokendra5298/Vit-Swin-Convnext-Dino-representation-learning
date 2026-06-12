import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="k-NN evaluation using extracted frozen features")
    parser.add_argument("--train-features", type=str, required=True)
    parser.add_argument("--test-features", type=str, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def load_features(path: str):
    data = np.load(path, allow_pickle=True)
    features = torch.from_numpy(data["features"]).float()
    labels = torch.from_numpy(data["labels"]).long()
    class_names = data["class_names"].tolist()
    return features, labels, class_names


@torch.no_grad()
def weighted_knn(train_features, train_labels, test_features, test_labels, num_classes, k, temperature, batch_size):
    train_features = torch.nn.functional.normalize(train_features, dim=1)
    test_features = torch.nn.functional.normalize(test_features, dim=1)
    total = 0
    correct_top1 = 0
    correct_top5 = 0

    for start in range(0, len(test_features), batch_size):
        end = min(start + batch_size, len(test_features))
        feats = test_features[start:end]
        labels = test_labels[start:end]
        sim = feats @ train_features.T
        top_sim, top_indices = sim.topk(k, dim=1)
        top_labels = train_labels[top_indices]
        weights = torch.exp(top_sim / temperature)
        probs = torch.zeros(feats.size(0), num_classes)
        probs.scatter_add_(1, top_labels, weights)
        _, preds = probs.topk(min(5, num_classes), dim=1)
        correct_top1 += (preds[:, 0] == labels).sum().item()
        correct_top5 += (preds == labels.view(-1, 1)).any(dim=1).sum().item()
        total += labels.size(0)

    return {
        "top1_percent": 100.0 * correct_top1 / max(1, total),
        "top5_percent": 100.0 * correct_top5 / max(1, total),
        "total": total,
    }


def main():
    args = parse_args()
    train_features, train_labels, class_names = load_features(args.train_features)
    test_features, test_labels, _ = load_features(args.test_features)

    train_mask = train_labels >= 0
    test_mask = test_labels >= 0
    train_features = train_features[train_mask]
    train_labels = train_labels[train_mask]
    test_features = test_features[test_mask]
    test_labels = test_labels[test_mask]

    metrics = weighted_knn(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        num_classes=len(class_names),
        k=args.k,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )
    metrics["k"] = args.k
    metrics["temperature"] = args.temperature
    print(json.dumps(metrics, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2))
        print(f"Saved k-NN metrics to: {output}")


if __name__ == "__main__":
    main()
