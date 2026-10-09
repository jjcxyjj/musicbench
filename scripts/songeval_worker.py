"""Run upstream SongEval in its own environment; no musicbench imports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = p.parse_args()
    import librosa
    import numpy as np
    import torch
    from muq import MuQ
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    repo = Path(args.repo)
    spec = importlib.util.spec_from_file_location("musicbench_upstream_songeval", repo / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = OmegaConf.to_container(OmegaConf.load(repo / "config.yaml"))["generator"]
    config.pop("_target_")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = repo / "ckpt/model.safetensors"
    model = module.Generator(**config)
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    model = model.to(device).eval()
    muq_name = "OpenMuQ/MuQ-large-msd-iter"
    muq = MuQ.from_pretrained(muq_name).to(device).eval()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results, errors = [], []
    dimensions = ["coherence", "musicality", "memorability", "clarity", "naturalness"]
    with torch.inference_mode():
        for row in rows:
            try:
                audio, _ = librosa.load(row["audio_path"], sr=24000, mono=True)
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError("Empty or non-finite audio")
                features = muq(torch.from_numpy(audio).unsqueeze(0).to(device),
                               output_hidden_states=True)["hidden_states"][6]
                scores = model(features).squeeze(0).cpu().tolist()
                if len(scores) != 5 or not np.isfinite(scores).all():
                    raise ValueError("Invalid model scores")
                values = dict(zip(dimensions, scores))
                # This dimension concerns vocal breathing; require explicit vocal metadata.
                if row.get("has_vocal") is not True:
                    values["naturalness"] = None
                results.append({"id": row["id"], **values})
            except Exception as exc:
                errors.append({"id": row["id"], "reason": str(exc)})
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = {"per_sample": results, "errors": errors,
              "model": {"repository": "https://github.com/ASLP-lab/SongEval",
                        "commit": commit, "checkpoint_sha256": digest,
                        "muq": muq_name, "device": device, "torch": torch.__version__}}
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
