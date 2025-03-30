import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        generated_tokens: List[int] = []

        # Ensure no gradients are computed during inference
        with torch.no_grad():
            for _ in range(self.max_output_len):
                # Forward pass: compute logits using only the LM head
                outputs = self.model(input_ids)
                logits  = outputs.logits #Shape:(1, seq_len, vocab_size)
                # 1 is the batch size, seq_len is the number of tokens in the input, vocab_size is the number of possible tokens in model's vocabulary

                # Get the logits for the last token; this isolates the probabilities for the next token
                last_token_logits = logits[:, -1, :]
                # greedily pick the token with the highest probability 
                next_token = torch.argmax(last_token_logits, dim = -1) # Shape: (1, )
                # next_token is a tensor of shape (1,) containing with the index of the most probable token

                # Append the predicted token to the generated tokens list
                generated_tokens.append(next_token.item()) # Stop if EOS token is generated
                if next_token.item() == self.eos_token_id:
                    break 
                # Append the generated token to the input_ids for the next step. 
                # Unsqueeze to match dimensions: (1,) -> (1,1)
                input_ids = torch.cat([input_ids,next_token.unsqueeze(0)],dim=1)
                # input_ids had shape (1,seq_len) and now has shape (1,seq_len+1). where seq_len is current sequence length 
                # next_token.unsqueeze(0) # Shape: (1,1) unsqueeze(0) adds a batch dimension, making the shape explicitly match input_ids 
        # return a tensor containing only the generated tokens
        return torch.tensor(generated_tokens, dtype = torch.int)
                



        # TODO:
        raise NotImplementedError

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

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
        # TODO:
        raise NotImplementedError
            