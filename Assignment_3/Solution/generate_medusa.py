import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

import torch.nn.functional as F

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
                
    def get_probability_distributions(self, input_ids):
        """
        Given input_ids, returns a list of probability distributions from 
        the LM head and S Medusa heads.
        """
        with torch.no_grad():
            outputs = self.model(input_ids, medusa_forward=True, output_orig=True)

            medusa_logits, _, lm_logits = outputs  # Unpack outputs
            lm_probs = F.softmax(lm_logits[:, -1, :], dim=-1)  # LM head probabilities

            # Get probabilities from Medusa heads
            medusa_probs = [F.softmax(medusa_logits[i][:, -1, :], dim=-1) for i in range(self.no_heads - 1)]

            return [lm_probs] + medusa_probs  # Return LM head + S Medusa heads

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
    # Since the function signature specifies a return type of Int[torch.Tensor, "batch out_seq_len"] (i.e., a 2D tensor with shape (batch, out_seq_len)), you need to modify it to return a tensor with shape (1, T) when batch size is 1. You can achieve this by unsqueezing the 0-th dimension:

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        """
            Implement multi-head decoding technique.

            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        """    
        device = input_ids.device
        candidates = [(input_ids.clone(), 0.0)]  # (sequence, score)
        
        for _ in range(self.max_output_len):
            new_candidates = []
            for seq, score in candidates:
                with torch.no_grad():
                    medusa_outputs = self.model(
                        seq, medusa_forward=True, output_orig=True
                    )
                    medusa_logits, _, lm_logits = medusa_outputs
                
                log_probs = F.log_softmax(lm_logits[:, -1, :], dim=-1)
                top_w_probs, top_w_tokens = torch.topk(log_probs, self.beam_width, dim=-1)
                
                for i in range(self.beam_width):
                    new_seq = torch.cat([seq, top_w_tokens[:, i].unsqueeze(-1)], dim=-1)
                    new_score = score + top_w_probs[:, i].item()
                    new_candidates.append((new_seq, new_score))
                
            # Keep Top-W candidates based on score
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:self.beam_width]
            
            # Stop if EOS token is found in all candidates
            if all(self.eos_token_id in seq for seq, _ in candidates):
                break
        
        best_sequence = max(candidates, key=lambda x: x[1])[0]
        return best_sequence.squeeze(0)
    
    def multi_head_decoding_erroneous(
        self,
        input_ids: torch.FloatTensor,  # Shape: (1, P)
    ) -> torch.IntTensor:
        '''
            Multi-head decoding with Medusa heads.

            - batch size is always 1.
            - stop decoding when:
                - EOS token is generated (`self.eos_token_id`)
                - max tokens (`self.max_output_len`) are generated.

            Returns:
                Tensor of shape (T,) where T <= self.max_output_len
        '''
        candidates = [input_ids]  # List of sequences to explore
        scores = torch.zeros(len(candidates), device=input_ids.device)  # Initial scores

        for _ in range(self.max_output_len):
            new_candidates = []
            new_scores = []

            with torch.inference_mode():
                print(candidates[-1].unsqueeze(0).shape)  # Before passing to head
                candidates[-1] = candidates[-1].unsqueeze(0)  # Add batch dim
                candidates[-1] = F.pad(candidates[-1], (0, 4096 - 183))  # Zero-pad to 4096
                medusa_logits = torch.stack(
                    [head(candidates[-1].unsqueeze(0).to(next(head.parameters()).dtype)) for head in self.model.medusa_head],
                    dim=0,
                ) # Shape: (num_heads, 1, P, vocab_size)

            for i, candidate in enumerate(candidates):
                # Get token probabilities from multiple heads
                log_probs = [F.log_softmax(medusa_logits[:, 0, -1, :], dim=-1)]  # Get last token logits

                for step in range(len(log_probs)):  # Loop over Medusa heads
                    top_w_tokens = torch.topk(log_probs[step], self.beam_width, dim=-1)

                    for j in range(self.beam_width):
                        print(top_w_tokens.indices[:, j].shape)
                        new_token = top_w_tokens.indices[:, j].squeeze().item()  # Get token ID
                        new_score = scores[i] + top_w_tokens.values[:, j].squeeze().item()
                        new_candidate = torch.cat([candidate, torch.tensor([[new_token]], device=candidate.device)], dim=1)

                        # Stop if EOS token is generated
                        if new_token == self.eos_token_id:
                            return new_candidate.squeeze(0)[input_ids.shape[1]:]

                        new_candidates.append(new_candidate)
                        new_scores.append(new_score)

            if new_candidates:
                # Posterior verification with Medusa confidence
                verified_candidates, verified_scores = [], []
                for i, candidate in enumerate(new_candidates):
                    if self.posterior_threshold is not None:
                        posterior_confidence = self.compute_posterior_confidence(candidate)
                        if posterior_confidence > self.posterior_threshold:
                            verified_candidates.append(candidate)
                            verified_scores.append(new_scores[i])
                    else:
                        verified_candidates.append(candidate)
                        verified_scores.append(new_scores[i])

                if verified_candidates:
                    # Select top-k candidates after verification
                    top_indices = torch.topk(torch.tensor(verified_scores), self.beam_width).indices
                    candidates = [verified_candidates[i] for i in top_indices]
                    scores = torch.tensor([verified_scores[i] for i in top_indices], device=input_ids.device)
                else:
                    break  # Stop if no valid candidates remain
            else:
                break  # Stop if no valid candidates remain

        # Return best candidate (excluding input tokens)
        best_candidate = candidates[torch.argmax(scores)]
        return best_candidate.squeeze(0)[input_ids.shape[1]:]