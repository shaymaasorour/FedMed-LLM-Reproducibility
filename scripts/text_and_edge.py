"""Section 3.1 (free-text statistics per client, RoBERTa-Large tokenizer) and Section 3.7 (trainable parameters,
upload size and peak memory of local LoRA fine-tuning; geometry read from each backbone's configuration)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer
from fedmed.config import CFG
from fedmed import analysis, llm, systems

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=".", help="folder with dataset_1.csv ... dataset_5.csv")
p.add_argument("--skip-text", action="store_true", help="only compute the edge-hardware table")
args = p.parse_args()

if not args.skip_text:
    llm.init_clients(args.data_dir, seed=42)
    tok = AutoTokenizer.from_pretrained(llm.MODEL_REGISTRY["RoBERTa-Large"])
    print("=== Free-text statistics per client (tokenizer: RoBERTa-Large BPE) ===\n")
    print(analysis.text_statistics(tok).to_string(index=False))

geometry = {"T5-Large (encoder)": systems.backbone_geometry("T5-Large"),
            "GLM-RoBERTa-Large": systems.backbone_geometry("GLM-RoBERTa"),
            "RoBERTa-Large": systems.backbone_geometry("RoBERTa-Large")}
Bc = CFG["stage_B_llm"]
print(f"\n=== Local LoRA fine-tuning cost per client (seq len {Bc['max_len']}, r = {Bc['lora_r']}) ===\n")
print(analysis.edge_cost(geometry).to_string(index=False))
