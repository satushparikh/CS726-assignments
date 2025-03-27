import torch
import torch.nn as nn
import warnings

from jaxtyping import Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

warnings.filterwarnings("ignore")

class TrieNode:
    """ A Trie node for storing tokenized words. """
    def __init__(self):
        self.children: Dict[int, TrieNode] = {}  # Maps token ID to TrieNode
        self.is_end_of_word = False  # Marks the end of a valid word

class Trie:
    """ A Trie data structure to store tokenized words efficiently. """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokenized_word: List[int]):
        """ Insert a tokenized word into the Trie. """
        node = self.root
        for token in tokenized_word:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_word = True  # Mark the end of a valid word
    
    def get_valid_next_tokens(self, prefix_tokens: List[int]) -> List[int]:
        node = self.root
        matched_prefix = []
        for token in prefix_tokens:
            if token in node.children:
                node = node.children[token]
                matched_prefix.append(token)
            else:
                break
        
        if not matched_prefix:
            return list(self.root.children.keys())

        return list(node.children.keys())

class ConstrainedTextGenerator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        eos_id: int,
        max_output_len: int = 10,
    ) -> None:
        """
        Initialize the ConstrainedTextGenerator class.

        model: LLM
        tokenizer: LLM's tokenizer.
        eos_id: End-of-sequence token id
        max_output_len: Maximum number of tokens to be generated.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_id
        self.max_output_len = max_output_len
    
    def build_trie(self, word_list: List[str]) -> Trie:
        """Builds a Trie from the given word list"""
        trie = Trie()
        for word in word_list:
            tokenized_word = self.tokenizer.encode(word, add_special_tokens=False)
            trie.insert(tokenized_word)
        return trie

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: List[str]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        """
        Implement Word-Constrained decoding technique.

        Parameters:
        - input_ids: Tensor of shape (1, P) (input prompt)
        - word_list: List of words that must appear in the generated output

        Returns:
        - Generated token tensor of shape (T,) where T ≤ max_output_len
        """
        trie = self.build_trie(word_list)

        generated_tokens = []
        current_input = input_ids  # Shape: (1, P)

        for step in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
                logits = outputs.logits[:, -1, :] # Shape: (1, seq_len, vocab_size)

            valid_tokens = trie.get_valid_next_tokens(generated_tokens)
            if valid_tokens:
                invalid_token_indices = [i for i in range(logits.shape[-1]) if i not in valid_tokens]
                logits[:, invalid_token_indices] = -10

            if step < 3 and valid_tokens:
                logits[:, valid_tokens] += 5

            probs = torch.nn.functional.softmax(logits, dim=-1)
            
#                print(f"Logits before modification {logits[:, invalid_token_indices]}")
#                logits[:, [i for i in range(logits.shape[-1]) if i not in valid_tokens]] -= 6.9
#                print(f"Logits after modification {logits[:, invalid_token_indices]}")

            next_token = torch.multinomial(probs, 1) # Shape: (1, 1)
            generated_tokens.append(next_token.item())
            if next_token.item() == self.eos_token_id:
                break

            current_input = torch.cat([current_input, next_token], dim=-1)
        return torch.tensor(generated_tokens, dtype=torch.long)
