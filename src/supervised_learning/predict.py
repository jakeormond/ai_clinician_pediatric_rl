"""
For generating predictions using a trained Dynamic-DeepHit model across the 
entire test dataset. Outputs predictions to a Parquet file along with metadata.

"""

# predict.py
import argparse
import json
import torch
import pandas as pd



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = DynamicDeepHit.from_config(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"], strict=True)

    dataset = TestTrajectoryDataset(args.test_data)

    df = run_predictions(model, dataset, batch_size=args.batch_size, device=args.device)

    # Optional: add health score post-processing or leave to downstream scripts
    # df["health_score"] =...

    df.to_parquet(args.output, index=False)

    meta = {
        "checkpoint": args.checkpoint,
        "config": ckpt.get("config", {}),
        "num_rows": len(df),
        "device": args.device,
    }
    with open(args.output + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()