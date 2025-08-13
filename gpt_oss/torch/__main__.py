#!/usr/bin/env python

import argparse
import sys
import torch

from gpt_oss.torch.model import TokenGenerator
from gpt_oss.tokenizer import get_tokenizer


def main(args):
    parser = argparse.ArgumentParser(
        description='Generate text with gpt-oss PyTorch model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model', metavar='PATH', type=str, help='Path to gpt-oss checkpoint directory (e.g., model_dir/)')
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompt text')
    parser.add_argument('-l', '--limit', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Sampling temperature (0 for greedy)')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu)')
    parser.add_argument('--show-tokens', action='store_true', help='Show token IDs')
    
    options = parser.parse_args(args)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    # Initialize model
    device = torch.device(options.device)
    generator = TokenGenerator(options.model, device)
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(options.prompt)
    if options.show_tokens:
        print(f"Prompt tokens: {prompt_tokens}")
    
    # Define stop tokens (end of text)
    stop_tokens = [tokenizer.encode("<|endoftext|>")[0]]
    
    # Generate tokens
    generated_text = ""
    for token in generator.generate(
        prompt_tokens=prompt_tokens,
        stop_tokens=stop_tokens,
        temperature=options.temperature,
        max_tokens=options.limit,
        return_logprobs=False
    ):
        text = tokenizer.decode([token])
        generated_text += text
        print(text, end='', flush=True)
    
    print()  # New line at the end


if __name__ == '__main__':
    main(sys.argv[1:])