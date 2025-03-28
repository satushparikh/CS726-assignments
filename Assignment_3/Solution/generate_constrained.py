import torch
import torch.nn as nn
import warnings

from jaxtyping import Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Set

warnings.filterwarnings("ignore")

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class ConstrainedTextGenerator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        eos_id: int,
        max_output_len: int = 10,
    ) -> None:
        self.model = model
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tokenizer = tokenizer

    def _build_trie_from_word_list(self, word_list: List[str]) -> TrieNode:
        """Build a trie from the given word list."""
        root = TrieNode()
        for word in word_list:
            # Tokenize the word and add to trie
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            node = root
            for token in tokens:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
            node.is_end_of_word = True
        return root

    def _get_valid_tokens(self, node: TrieNode) -> Set[int]:
        """Get all valid tokens from the current trie node."""
        return set(node.children.keys())

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        trie_root = self._build_trie_from_word_list(word_list)
        current_trie_node = trie_root

        current_inputs = input_ids
        
        generated_tokens = []
        past_key_values = None
        
        # Ensure input_ids has shape [1, seq_len]
        if current_inputs.dim() == 1:
            current_inputs = current_inputs.unsqueeze(0)
            
        for _ in range(self.max_output_len):
            model_inputs = {
                "input_ids": current_inputs,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True
            }
            
            outputs = self.model(**model_inputs)
            
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            valid_tokens = self._get_valid_tokens(current_trie_node)
            if not valid_tokens:
                break
                
            # Apply mask to logits: set '-inf' for invalid tokens
            mask = torch.ones_like(next_token_logits, device=next_token_logits.device) * float('-inf')
            for token in valid_tokens:
                mask[0, token] = 0
            masked_logits = next_token_logits + mask
            
            next_token = torch.argmax(masked_logits, dim=-1, keepdim=True)
            
            if next_token.item() == self.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            current_inputs = next_token
            
            # Update trie node
            current_trie_node = current_trie_node.children[next_token.item()]
                
            # If we've completed a word, reset to root of trie
            if current_trie_node.is_end_of_word:
                current_trie_node = trie_root
        
        return torch.tensor(generated_tokens, dtype=torch.long)
