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

class TokenTrie:
    def __init__(
            self, 
            eos_token_id
        ) -> None:
        '''
            Initialize the TokenTrie class.
            eos_token_id: End-of-sequence token id 
            Do not edit.
        '''
        self.root = TrieNode()
        self.eos_token_id = eos_token_id

    def insert(
            self, 
            token_sequence
        ) -> None:
        '''
            Inserts a token into the TokenTrie.
            token_sequence: A list of tokens to be inserted as a sequence

            This method traverses the trie and adds node for each token
            It also marks the last token as end of word
        '''
        node = self.root
        for token in token_sequence:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_word = True  # Mark as complete word

    def get_next_tokens(
            self, 
            node
        ) -> None:
        '''
            Get the next possible tokens from a given node.
            node: Current node in the Trie
        '''
        return set(node.children.keys())

    def is_complete_word(
            self, 
            token_sequence
        ) -> None:
        '''
            Check if a given token sequence forms a complete word in the Trie.
            
            token_sequence: A list of tokens to check.
            
            Returns True if the sequence forms a complete word in the Trie, 
            otherwise returns False.
        '''
        node = self.root
        for token in token_sequence:
            if token not in node.children:
                return False
            node = node.children[token]
        return node.is_end_of_word

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_id
        self.max_output_len = max_output_len

    def _build_trie_from_word_list(self, word_list: List[str]) -> TokenTrie:
        trie = TokenTrie(self.eos_token_id)
        
        for word in word_list:
            tokens = self.tokenizer.tokenize(' ' + word)  # Space ensures proper tokenization
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            trie.insert(token_ids)
        
        trie.insert([self.eos_token_id])  # Ensure EOS is in trie
        return trie

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # Build Trie and get the parent node
        trie = self._build_trie_from_word_list(word_list)
        current_trie_node = trie.root

        generated_tokens = []
        current_prefix = []
        device = input_ids.device

        for _ in range(self.max_output_len):
            # Prepare input for model
            if generated_tokens:
                input_with_generated = torch.cat([
                    input_ids,
                    torch.tensor(generated_tokens, device=device, dtype=torch.long).unsqueeze(0)
                ], dim=-1)
            else:
                input_with_generated = input_ids

            # Model prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_with_generated)
                next_token_logits = outputs.logits[0, -1, :]

            # Get allowed tokens from trie
            if current_prefix:
                allowed_tokens = trie.get_next_tokens(current_trie_node)
            else:
                allowed_tokens = trie.get_next_tokens(trie.root)

            if not allowed_tokens:
                break  # No valid tokens left

            # Apply mask
            mask = torch.full_like(next_token_logits, float('-inf'))
            mask[list(allowed_tokens)] = 0
            filtered_logits = next_token_logits + mask

            # Greedy decoding
            next_token = torch.argmax(filtered_logits).item()
            generated_tokens.append(next_token)

            # Update Trie state
            if next_token in current_trie_node.children:
                current_trie_node = current_trie_node.children[next_token]
                current_prefix.append(next_token)

            # If complete word, reset prefix & trie node
            if current_trie_node.is_end_of_word:
                if next_token == self.eos_token_id:
                    break  # Stop at EOS
                current_prefix = []  # Reset prefix
                current_trie_node = trie.root  # Reset to root for next word

        return torch.tensor(generated_tokens)