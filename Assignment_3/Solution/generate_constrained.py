import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

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
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

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
        word_token_ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in word_list]
        word_token_ids = {tuple(w): False for w in word_token_ids}

        generated_tokens = []
        current_input = input_ids
        used_words = set()

        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
                logits = outputs.logits

            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            missing_words = [tokens for tokens, used in word_token_ids.items() if not used]

            if missing_words:
                all_missing_tokens = {tok for word in missing_words for tok in word}
                logits[:, list(all_missing_tokens)] += 5.0 # Artificially boosting probabilities to select the missing words

            next_token = torch.argmax(probs, dim=-1)

            if next_token.item() == self.eos_token_id and all(word_token.values()):
                break

            generated_tokens.append(next_token)
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)

            for word_token in word_token_ids.keys():
                if len(generated_tokens) >= len(word_token) and tuple(generated_tokens[-len(word_token):]) == word_token:
                    word_token_ids[word_token] = True

            return torch.tensor(generated_tokens, dtype=torch.long)
        
