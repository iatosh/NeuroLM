"""
Refactored inference script for NeuroLM with YAML configuration
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import yaml
import torch
import numpy as np
import tiktoken
import matplotlib.pyplot as plt

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from downstream_dataset import HMCLoader


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {}).get('console', {})
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    return logging.getLogger(__name__)


@dataclass
class InferenceResults:
    """Container for inference results"""
    sample_idx: int
    true_label: str
    input_prompt: str
    generated_response: str
    inference_time: float
    model_path: str
    device: str
    eeg_shape: List[int]
    n_channels: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'sample_idx': self.sample_idx,
            'true_label': self.true_label,
            'input_prompt': self.input_prompt,
            'generated_response': self.generated_response,
            'inference_time': self.inference_time,
            'model_path': self.model_path,
            'device': self.device,
            'eeg_shape': self.eeg_shape,
            'n_channels': self.n_channels
        }


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _validate_config(self) -> None:
        """Validate configuration structure"""
        required_sections = ['model', 'architecture', 'dataset', 'generation']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
                
        # Validate paths
        model_config = self.config['model']
        if not Path(model_config['checkpoint_path']).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_config['checkpoint_path']}")
            
        if not Path(model_config['tokenizer_path']).exists():
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {model_config['tokenizer_path']}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value


class ModelManager:
    """Manages model loading and initialization"""
    
    def __init__(self, config: ConfigManager, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def load_model(self, device: str) -> NeuroLM:
        """Load and initialize NeuroLM model"""
        self.logger.info("Loading NeuroLM model...")
        
        # Get architecture config
        arch_config = self.config.get('architecture', {})
        
        # Initialize GPT config
        gpt_config = GPTConfig(
            n_layer=arch_config['n_layer'],
            n_head=arch_config['n_head'],
            n_embd=arch_config['n_embd'],
            block_size=arch_config['block_size'],
            bias=arch_config['bias'],
            vocab_size=arch_config['vocab_size'],
            dropout=arch_config['dropout']
        )
        
        # Initialize NeuroLM
        model = NeuroLM(
            GPT_config=gpt_config,
            tokenizer_ckpt_path=self.config.get('model.tokenizer_path'),
            init_from=self.config.get('model.init_from'),
            n_embd=arch_config['n_embd'],
            eeg_vocab_size=arch_config['eeg_vocab_size']
        )
        
        # Load checkpoint
        checkpoint_path = self.config.get('model.checkpoint_path')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = self._clean_state_dict(checkpoint['model'])
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Optional: Compile model for faster inference
        if self.config.get('hardware.compile', False) and hasattr(torch, 'compile'):
            self.logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model)
        
        self.logger.info(f"Model loaded successfully! Parameters: {model.get_num_params():,}")
        return model
        
    @staticmethod
    def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove unwanted prefixes from state dict"""
        unwanted_prefix = '_orig_mod.'
        cleaned = {}
        
        for k, v in state_dict.items():
            if k.startswith(unwanted_prefix):
                cleaned[k[len(unwanted_prefix):]] = v
            else:
                cleaned[k] = v
                
        return cleaned


class DataManager:
    """Manages dataset loading and processing"""
    
    def __init__(self, config: ConfigManager, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def load_dataset(self) -> HMCLoader:
        """Load HMC dataset"""
        dataset_config = self.config.get('dataset', {})
        data_path = Path(dataset_config['data_path'])
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_path}")
            
        files = os.listdir(data_path)
        if not files:
            raise ValueError(f"No files found in dataset directory: {data_path}")
            
        self.logger.info(f"Loading HMC dataset from {data_path}")
        
        dataset = HMCLoader(
            data_path,
            files=files,
            is_instruct=dataset_config.get('is_instruct', True),
            is_val=dataset_config.get('is_val', True),
            eeg_max_len=dataset_config.get('eeg_max_len', 512),
            text_max_len=dataset_config.get('text_max_len', 128)
        )
        
        self.logger.info(f"Dataset loaded: {len(dataset)} samples")
        return dataset
        
    def get_sample(self, dataset: HMCLoader, sample_idx: int) -> Dict[str, Any]:
        """Get a specific sample from dataset"""
        if sample_idx >= len(dataset):
            raise ValueError(f"Sample index {sample_idx} out of range (max: {len(dataset)-1})")
            
        # Get data tuple
        (eeg_data, text_prompt, label, input_chans, 
         input_time, eeg_mask, gpt_mask) = dataset[sample_idx]
        
        # HMC channel names
        channel_names = ['F4', 'C4', 'O2', 'C3']
        
        return {
            'eeg_data': eeg_data,
            'text_prompt': text_prompt,
            'label': label,
            'input_chans': input_chans,
            'input_time': input_time,
            'eeg_mask': eeg_mask,
            'gpt_mask': gpt_mask,
            'channel_names': channel_names
        }


class Visualizer:
    """Handles data visualization"""
    
    def __init__(self, config: ConfigManager, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.viz_config = config.get('visualization', {})
        
    def visualize_eeg(
        self, 
        eeg_data: np.ndarray, 
        channel_names: List[str], 
        output_path: Path
    ) -> None:
        """Create and save EEG visualization"""
        n_channels = len(channel_names)
        figsize = self.viz_config.get('figsize_per_channel', [12, 2])
        
        fig, axes = plt.subplots(
            n_channels, 1, 
            figsize=(figsize[0], figsize[1] * n_channels), 
            sharex=True
        )
        
        if n_channels == 1:
            axes = [axes]
            
        # Time axis
        sampling_rate = self.viz_config.get('sampling_rate', 200)
        time = np.arange(eeg_data.shape[1]) / sampling_rate
        
        for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
            ax.plot(time, eeg_data[i, :], 'b-', linewidth=0.5)
            ax.set_ylabel(ch_name)
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('EEG Signal Visualization', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = self.viz_config.get('dpi', 150)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"EEG visualization saved to {output_path}")
        
    @staticmethod
    def reshape_eeg_for_visualization(
        eeg_data: torch.Tensor,
        eeg_mask: torch.Tensor,
        n_channels: int,
        samples_per_patch: int
    ) -> np.ndarray:
        """Reshape EEG data for visualization"""
        eeg_numpy = eeg_data.numpy()
        
        # Get valid data length
        valid_rows = int(eeg_mask.sum().item())
        eeg_valid = eeg_numpy[:valid_rows, :]
        
        # Calculate dimensions
        num_time_segments = valid_rows // n_channels
        
        # Reshape
        eeg_temp = eeg_valid.reshape(num_time_segments, n_channels, samples_per_patch)
        eeg_permuted = eeg_temp.transpose(1, 0, 2)
        eeg_final = eeg_permuted.reshape(n_channels, -1)
        
        return eeg_final


class InferenceEngine:
    """Handles model inference"""
    
    def __init__(self, model: NeuroLM, config: ConfigManager, device: str, logger: logging.Logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
    def run(self, sample_data: Dict[str, Any]) -> InferenceResults:
        """Run inference on a sample"""
        # Decode input prompt
        input_text = self._decode_tokens(sample_data['text_prompt'].tolist())
        self.logger.info(f"Input prompt: {input_text}")
        
        # Prepare batch
        batch_data = self._prepare_batch(sample_data)
        
        # Generate
        gen_config = self.config.get('generation', {})
        
        start_time = time.time()
        with torch.no_grad():
            generated = self.model.generate(
                x_eeg=batch_data['eeg'],
                x_text=batch_data['text'],
                input_chans=batch_data['chans'],
                input_time=batch_data['time'],
                input_mask=batch_data['mask'],
                max_new_tokens=gen_config.get('max_new_tokens', 30),
                temperature=gen_config.get('temperature', 1.0),
                top_k=gen_config.get('top_k', 50)
            )
        inference_time = time.time() - start_time
        
        # Decode output
        generated_text = self._decode_tokens(generated[0].tolist())
        self.logger.info(f"Generated response: {generated_text}")
        self.logger.info(f"Inference time: {inference_time:.3f}s")
        
        # Get true label
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        true_label = label_map.get(sample_data['label'], 'Unknown')
        
        return InferenceResults(
            sample_idx=self.config.get('dataset.sample_idx'),
            true_label=true_label,
            input_prompt=input_text,
            generated_response=generated_text,
            inference_time=inference_time,
            model_path=self.config.get('model.checkpoint_path'),
            device=self.device,
            eeg_shape=list(sample_data['eeg_data'].shape),
            n_channels=len(sample_data['channel_names']),
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
    def _prepare_batch(self, sample_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare batch data for model"""
        return {
            'eeg': sample_data['eeg_data'].unsqueeze(0).to(self.device),
            'text': sample_data['text_prompt'].unsqueeze(0).to(self.device),
            'chans': sample_data['input_chans'].unsqueeze(0).to(self.device),
            'time': sample_data['input_time'].unsqueeze(0).to(self.device),
            'mask': sample_data['eeg_mask'].unsqueeze(0).to(self.device)
        }
        
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens to text"""
        valid_tokens = [t for t in tokens if t < 50257]
        return self.tokenizer.decode(valid_tokens)


class WandbLogger:
    """Optional Weights & Biases integration"""
    
    def __init__(self, config: ConfigManager, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.wandb_config = config.get('logging.wandb', {})
        self.enabled = self.wandb_config.get('enabled', False) and WANDB_AVAILABLE
        
    def init(self, device: str) -> None:
        """Initialize wandb run"""
        if not self.enabled:
            return
            
        wandb.init(
            project=self.wandb_config.get('project', 'NeuroLM_Inference'),
            name=self.wandb_config.get('run_name', 'inference_run'),
            config={
                'model_path': self.config.get('model.checkpoint_path'),
                'device': device,
                'config': self.config.config
            }
        )
        self.logger.info(f"Wandb initialized: {wandb.run.name}")
        
    def log(self, data: Dict[str, Any]) -> None:
        """Log data to wandb"""
        if self.enabled:
            wandb.log(data)
            
    def log_image(self, name: str, path: Path) -> None:
        """Log image to wandb"""
        if self.enabled:
            wandb.log({name: wandb.Image(str(path))})
            
    def finish(self) -> None:
        """Finish wandb run"""
        if self.enabled:
            wandb.finish()


class InferenceOrchestrator:
    """Main orchestrator for the inference pipeline"""
    
    def __init__(self, config_path: str):
        # Load configuration
        self.config = ConfigManager(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config.config)
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize components
        self.model_manager = ModelManager(self.config, self.logger)
        self.data_manager = DataManager(self.config, self.logger)
        self.visualizer = Visualizer(self.config, self.logger)
        self.wandb_logger = WandbLogger(self.config, self.logger)
        
        # Setup output directory
        self.output_dir = self._setup_output_dir()
        
    def run(self) -> None:
        """Execute the complete inference pipeline"""
        try:
            # Initialize wandb
            self.wandb_logger.init(self.device)
            
            # Load model
            model = self.model_manager.load_model(self.device)
            
            # Load dataset
            dataset = self.data_manager.load_dataset()
            
            # Get sample
            sample_idx = self.config.get('dataset.sample_idx', 0)
            sample_data = self.data_manager.get_sample(dataset, sample_idx)
            
            # Visualize EEG if enabled
            if self.config.get('output.save_visualization', True):
                self._visualize_sample(sample_data)
                
            # Run inference
            engine = InferenceEngine(model, self.config, self.device, self.logger)
            results = engine.run(sample_data)
            
            # Save results
            if self.config.get('output.save_json', True):
                self._save_results(results)
                
            # Log to wandb
            self._log_results(results)
            
            # Print memory usage
            self._print_memory_usage()
            
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}", exc_info=True)
            raise
            
        finally:
            self.wandb_logger.finish()
            
    def _setup_device(self) -> str:
        """Setup compute device"""
        device_config = self.config.get('hardware.device', 'auto')
        
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
            
        if device == 'cuda' and torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("Using CPU")
            
        return device
        
    def _setup_output_dir(self) -> Path:
        """Create output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.config.get('output.base_dir', 'results'))
        output_dir = base_dir / f"inference_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
        
    def _visualize_sample(self, sample_data: Dict[str, Any]) -> None:
        """Visualize EEG data"""
        # Reshape data
        eeg_reshaped = self.visualizer.reshape_eeg_for_visualization(
            sample_data['eeg_data'],
            sample_data['eeg_mask'],
            len(sample_data['channel_names']),
            sample_data['eeg_data'].shape[1]
        )
        
        # Limit samples
        max_samples = self.config.get('visualization.max_samples', 1000)
        viz_data = eeg_reshaped[:, :max_samples]
        
        # Save visualization
        viz_path = self.output_dir / "eeg_visualization.png"
        self.visualizer.visualize_eeg(
            viz_data,
            sample_data['channel_names'],
            viz_path
        )
        
        # Log to wandb
        self.wandb_logger.log_image("eeg_visualization", viz_path)
        
    def _save_results(self, results: InferenceResults) -> None:
        """Save results to JSON"""
        results_path = self.output_dir / 'results.json'
        
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
            
        self.logger.info(f"Results saved to {results_path}")
        
    def _log_results(self, results: InferenceResults) -> None:
        """Log results to wandb"""
        self.wandb_logger.log({
            'inference/sample_idx': results.sample_idx,
            'inference/true_label': results.true_label,
            'inference/inference_time_sec': results.inference_time,
            'model/n_channels': results.n_channels,
        })
        
    def _print_memory_usage(self) -> None:
        """Print GPU memory usage"""
        if self.device == 'cuda':
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            current_memory = torch.cuda.memory_allocated() / 1024**3
            
            self.logger.info(f"GPU memory - Current: {current_memory:.2f} GB, Max: {max_memory:.2f} GB")
            
            self.wandb_logger.log({
                'system/max_gpu_memory_gb': max_memory,
                'system/current_gpu_memory_gb': current_memory,
            })


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NeuroLM Inference with YAML Configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference_config.yaml',
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--sample_idx',
        type=int,
        help='Override sample index from config'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help='Override device from config'
    )
    
    args = parser.parse_args()
    
    # Load and potentially override config
    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
        
    # Create temporary config with overrides if needed
    if args.sample_idx is not None or args.device is not None:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        if args.sample_idx is not None:
            config_dict['dataset']['sample_idx'] = args.sample_idx
            
        if args.device is not None:
            config_dict['hardware']['device'] = args.device
            
        # Save temporary config
        temp_config_path = Path('configs/.temp_inference_config.yaml')
        temp_config_path.parent.mkdir(exist_ok=True)
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_dict, f)
            
        config_path = str(temp_config_path)
    
    # Run inference
    orchestrator = InferenceOrchestrator(config_path)
    orchestrator.run()
    
    # Cleanup temp config if created
    if 'temp_config_path' in locals():
        temp_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()