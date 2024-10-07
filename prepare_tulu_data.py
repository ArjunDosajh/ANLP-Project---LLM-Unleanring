import logging
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
from rich.progress import track
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

def main(opts) -> None:
    # Load the tokenizer using AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(opts.tokenizer)

    # Load your DataFrame
    df = pd.read_csv(opts.input_csv)  # Assuming your data is in a CSV file

    log.info("Tokenizing dataset...")
    tokenized_data = [preprocess(row, tokenizer, opts.seq_len) for _, row in track(df.iterrows())]

    log.info("Filtering dataset...")
    filtered_data = [item for item in tokenized_data if item['n_labels'] > 0]
    log.info(f"Filtered out {len(tokenized_data) - len(filtered_data)} examples")

    log.info("Counting tokens...")
    total_tokens = sum(len(item['input_ids']) for item in filtered_data)
    log.info(f"Total tokens: {total_tokens:,d}")

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"), dtype=np.uint16, mode="w+", shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
    )
    offset = 0
    for item in track(filtered_data):
        ex_len = len(item["input_ids"])
        input_ids_file[offset : offset + ex_len] = item["input_ids"]
        label_mask_file[offset : offset + ex_len] = item["label_mask"]
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")

def preprocess(row, tokenizer, max_seq_len: int):
    if row['task'] == 'Task2':  # Question Answering
        input_text = f"<|human|>\n{row['input']}\n<|assistant|>\n{row['output']}"
    else:  # Sentence Completion
        input_text = f"<|human|>\n{row['input']}\n<|assistant|>\n{row['output']}"

    tokens = tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=max_seq_len)
    input_ids = tokens

    # Create label mask: False for input, True for output
    split_point = input_text.index("<|assistant|>")
    input_length = len(tokenizer.encode(input_text[:split_point], add_special_tokens=True))
    label_mask = [False] * input_length + [True] * (len(tokens) - input_length)

    # Truncate or pad to max_seq_len
    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare custom dataset for OLMo fine-tuning")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save the results to")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="Tokenizer identifier",
        default="allenai/OLMo-7B-0724-Instruct"
    )
    parser.add_argument("-s", "--seq-len", type=int, help="Max sequence length", default=2048)
    return parser

if __name__ == "__main__":
    opts = get_parser().parse_args()
    main(opts)