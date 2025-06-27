"""
Simple inference script for NeuroLM
Demonstrates basic model functionality with a single EEG sample
"""

import os
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import tiktoken
from datetime import datetime
import wandb

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from downstream_dataset import HMCLoader


def load_model(model_path, tokenizer_path, device='cuda'):
    """Load NeuroLM model from checkpoint"""
    print("Loading model...")
    
    # Initialize GPT config
    gpt_config = GPTConfig(
        n_layer=12,
        n_head=12, 
        n_embd=768,
        block_size=1024,
        bias=False,
        vocab_size=50304,
        dropout=0.0
    )
    
    # Initialize NeuroLM
    model = NeuroLM(
        GPT_config=gpt_config,
        tokenizer_ckpt_path=tokenizer_path,
        init_from='gpt2',
        n_embd=768,
        eeg_vocab_size=8192
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model']
    
    # Remove unwanted prefix if exists
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully! Parameters: {model.get_num_params():,}")
    return model


def visualize_eeg(eeg_data, channel_names, output_path):
    """Visualize EEG signals"""
    n_channels = len(channel_names)
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    # Time axis (assuming 200 Hz sampling rate)
    time = np.arange(eeg_data.shape[1]) / 200.0
    
    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ax.plot(time, eeg_data[i, :], 'b-', linewidth=0.5)
        ax.set_ylabel(ch_name)
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('EEG Signal Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"EEG visualization saved to {output_path}")


def decode_tokens(tokens, tokenizer):
    """Decode tokens to text"""
    # Filter out special tokens and decode
    valid_tokens = [t for t in tokens if t < 50257]
    return tokenizer.decode(valid_tokens)


def run_inference(model, eeg_data, text_prompt, input_chans, input_time, input_mask, device='cuda'):
    """Run inference on a single EEG sample"""
    with torch.no_grad():
        # Move data to device
        eeg_data = eeg_data.unsqueeze(0).to(device)  # Add batch dimension
        text_prompt = text_prompt.unsqueeze(0).to(device)
        input_chans = input_chans.unsqueeze(0).to(device)
        input_time = input_time.unsqueeze(0).to(device)
        input_mask = input_mask.unsqueeze(0).to(device)
        
        # Generate text
        start_time = time.time()
        generated = model.generate(
            x_eeg=eeg_data,
            x_text=text_prompt,
            input_chans=input_chans,
            input_time=input_time,
            input_mask=input_mask,
            max_new_tokens=30,
            temperature=1.0,
            top_k=50
        )
        inference_time = time.time() - start_time
        
    return generated[0], inference_time  # Return first sample


def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Initialize wandb if requested
    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_path": args.model_path,
                "tokenizer_path": args.tokenizer_path,
                "data_path": args.data_path,
                "sample_idx": args.sample_idx,
                "device": device,
            }
        )
        print(f"Wandb initialized: {wandb.run.name}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"inference_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, args.tokenizer_path, device)

    # Load dataset
    print("\nLoading HMC dataset...")
    dataset = HMCLoader(
        Path(args.data_path),
        files=os.listdir(Path(args.data_path)), # HMCLoader expects a list of files
        is_instruct=True,
        is_val=True,
        eeg_max_len=512,
        text_max_len=128
    )

    # Get a single sample
    print(f"Dataset size: {len(dataset)} samples")
    sample_idx = args.sample_idx
    eeg_data, text_prompt, label, input_chans, input_time, eeg_mask, gpt_mask = dataset[sample_idx]
    # Get channel names (hardcoded for HMC dataset as channels are consistent)
    channel_names = [name.split('-')[0].strip() for name in ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']]

    print(f"\nEEG channels ({len(channel_names)}): {channel_names}")

    # Visualize EEG data
    print("\nVisualizing EEG data...")
    eeg_numpy = eeg_data.numpy()

    # Reshape for visualization
    # eeg_data (X_eeg from HMCLoader) is (eeg_max_len, samples_per_patch) where samples_per_patch is 200.
    # The actual data is in the first `valid_eeg_rows` rows, where `valid_eeg_rows` = `num_time_segments * n_channels`.
    # We need to reshape it to (n_channels, total_samples_actual).

    n_channels = len(channel_names) # 4 for HMC
    samples_per_patch = eeg_data.shape[1] # 200

    # Determine the actual length of valid EEG data rows using eeg_mask
    valid_eeg_rows = int(eeg_mask.sum().item()) # This is num_time_segments * n_channels
    eeg_numpy_actual = eeg_numpy[:valid_eeg_rows, :] # Extract only the valid data part (num_time_segments * n_channels, samples_per_patch)

    num_time_segments = valid_eeg_rows // n_channels # e.g., 120 // 4 = 30

    # Reshape from (num_time_segments * n_channels, samples_per_patch)
    # to (num_time_segments, n_channels, samples_per_patch)
    eeg_reshaped_temp = eeg_numpy_actual.reshape(num_time_segments, n_channels, samples_per_patch)
    
    # Permute to (n_channels, num_time_segments, samples_per_patch)
    eeg_reshaped_permuted = eeg_reshaped_temp.transpose(1, 0, 2) # (n_channels, num_time_segments, samples_per_patch)

    # Flatten the time segments to get (n_channels, total_samples_actual)
    eeg_reshaped = eeg_reshaped_permuted.reshape(n_channels, num_time_segments * samples_per_patch)
    
    eeg_viz_path = output_dir / "eeg_visualization.png"
    visualize_eeg(eeg_reshaped[:, :1000], channel_names, eeg_viz_path)
    
    # Log EEG visualization to wandb
    if args.wandb_log:
        wandb.log({"eeg_visualization": wandb.Image(str(eeg_viz_path))})
    
    # Run inference
    print("\nRunning inference...")
    print(f"Input EEG shape: {eeg_data.shape}")
    print(f"Input text prompt shape: {text_prompt.shape}")
    
    # Initialize tokenizer for decoding
    enc = tiktoken.get_encoding("gpt2")
    
    # Decode input prompt
    input_text = decode_tokens(text_prompt.tolist(), enc)
    print(f"\nInput prompt: {input_text}")
    
    # Generate response
    generated_tokens, inference_time = run_inference(
        model, eeg_data, text_prompt, input_chans, input_time, eeg_mask, device
    )
    
    # Decode generated text
    generated_text = decode_tokens(generated_tokens.tolist(), enc)
    print(f"\nGenerated response: {generated_text}")
    print(f"Inference time: {inference_time:.3f} seconds")
    
    # Get ground truth label
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'} 
    true_label = label_map.get(label, 'Unknown')
    print(f"True label: {true_label}")
    
    # Save results
    results = {
        'timestamp': timestamp,
        'sample_idx': sample_idx,
        'true_label': true_label,
        'input_prompt': input_text,
        'generated_response': generated_text,
        'inference_time': inference_time,
        'model_path': str(args.model_path),
        'device': device,
        'eeg_shape': list(eeg_data.shape),
        'n_channels': n_channels
    }
    
    # Log to wandb
    if args.wandb_log:
        wandb.log({
            'inference/sample_idx': sample_idx,
            'inference/true_label': true_label,
            'inference/inference_time_sec': inference_time,
            'inference/generated_tokens': len(generated_tokens),
            'model/n_channels': n_channels,
            'model/eeg_length': eeg_data.shape[0],
            'model/text_prompt_length': len(text_prompt),
        })
        
        # Log text results as a table
        wandb.log({
            'results': wandb.Table(
                columns=['Sample Index', 'True Label', 'Input Prompt', 'Generated Response', 'Inference Time (s)'],
                data=[[sample_idx, true_label, input_text, generated_text, f"{inference_time:.3f}"]]
            )
        })
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print memory usage
    if device == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nMax GPU memory used: {max_memory:.2f} GB")
        
        # Log GPU memory to wandb
        if args.wandb_log:
            wandb.log({
                'system/max_gpu_memory_gb': max_memory,
                'system/gpu_name': torch.cuda.get_device_name(0)
            })
    
    # Log final summary to wandb
    if args.wandb_log:
        wandb.summary['sample_idx'] = sample_idx
        wandb.summary['true_label'] = true_label
        wandb.summary['inference_time'] = inference_time
        wandb.summary['model_params'] = model.get_num_params()
        wandb.summary['output_dir'] = str(output_dir)
        
        # Finish wandb run
        wandb.finish()
        print("\nWandb run finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple NeuroLM inference')
    parser.add_argument('--model_path', type=str, default='checkpoints/NeuroLM-B.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='checkpoints/VQ.pt',
                        help='Path to tokenizer checkpoint')
    parser.add_argument('--data_path', type=str, default='/home/atosh/Research/Datasets/HMC/',
                        help='Path to HMC dataset directory')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of sample to use')
    parser.add_argument('--wandb_log', default=False, action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='NeuroLM_Inference',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='inference_run',
                        help='Wandb run name')
    
    args = parser.parse_args()
    main(args)