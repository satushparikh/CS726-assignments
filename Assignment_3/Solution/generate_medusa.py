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
        return torch.tensor(generated_tokens, dtype = torch.int).unsqueeze(0)
    # Since the function signature specifies a return type of Int[torch.Tensor, "batch out_seq_len"] (i.e., a 2D tensor with shape (batch, out_seq_len)), you need to modify it to return a tensor with shape (1, T) when batch size is 1. You can achieve this by unsqueezing the 0-th dimension:
                



        # TODO:
        # raise NotImplementedError

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
        generated_tokens: List[int] = []
        #Total number of heads: LM head(1) + S Medusa heads
        total_heads = self.no_heads # use_no_medusa_heads + 1
        # Continue generation until max_output_len tokens have been generated
        while len(generated_tokens) < self.max_output_len:
            with torch.no_grad():
                # STEP 1: Forward pass to obtain the probability distributions for the LM head and medusa heads
                outputs = self.model(input_ids)
                # LM head: obtain logits for the last token
                lm_logits = outputs.logits[:, -1 ,:] #shape: (1, vocab_size)
                p_list = [lm_logits]
                ##     !!!! Check this 
                # For each medusa head (assumed stored in ouputs.medusa_logits, a list of tensors)
                for k in range(total_heads - 1):
                    # Get the logits for the kth medusa head 
                    medusa_logits = outputs.medusa_logits[k][:, -1, :]
                    # Append the logits to the list
                    p_list.append(medusa_logits)

            # Step 2: Beam search over the next S+1 tokens
            # Initialize beam with one candidate: an empty list of new tokens and a score of 0.0
            candidates: List[List[int]] = [[]]
            scores: List[float] = [0.0]
            # {p_t, p_t+1, ..., p_t+S} = p_list
            # For each of the S+1 heads, extend the candidates 
            for s in range(len(p_list)):
                # compute log softmax for numerical stability
                log_probs = torch.log_softmax(p_list[s], dim =-1) # applies softmax along last dimension shape: (1, vocab_size)
                new_candidates = List[List[int]] = []
                new_scores = List[float] = []
                # for each candidate in the beam
                for cand, cand_score in zip(candidates, scores):
                    # Retrieve top W tokens from the current head's distribution 
                    top_log_probs, top_indices = torch.topk(log_probs[0], self.beam_width)
                    for i in range(self.beam_width):
                        token = top_indices[i].item()
                        token_log_prob =  top_log_probs[i].item()
                        # create a new candidate sequence by appending the token
                        new_candidates.append(cand+[token])
                        new_scores.append(cand_score + token_log_prob)
                # Retain only the top W candidates based on the new scores
                scores_tensor = torch.tensor(new_scores)
                sorted_scores, sorted_indices = torch.sort(scores_tensor, descending=True)
                top_indices = sorted_indices[:self.beam_width].tolist()
                candidates = [new_candidates[i] for i in top_indices]
                scores = [new_scores[i] for i in top_indices]
                
            # Step 3: Re-score candidate sequences using the LM head
            """ 
            Use the LM head to compute scores for all candidate sequences and pick the one with the highest score. Specifically, for a candidate sequence {ŷ_1,...,ŷ_t-1,ŷ_t,...,ŷ_t+S}, the score is computed as \sigma_{i=t}^{t+S} log p_i(ŷ_i | ŷ_1,...,ŷ_{i-1}) where log p_i is log_softmax(ŷ_i)
            """
            best_candidate = candidates[0]
            best_score = scores[0]
            for cand, score in zip(candidates, scores):
                if score > best_score:
                    best_candidate = cand
                    best_score = score

            # Optional: You can check for EOS token and trim the candidate if necessary.
            # Here we assume the candidate is used as is.
            best_candidate_tensor = torch.tensor(best_candidate, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            return best_candidate_tensor
        # TODO:
        # raise NotImplementedError
            