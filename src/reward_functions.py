# reward_functions.py
import re
import numpy as np
import torch
import traceback
from typing import Any, List, Dict, Union

from config import FP_TYPE, EPS, REWARD_CONFIG as global_reward_config
from reward_globals import (
    DEVICE, nli_tokenizer, nli_model, semantic_model,
    cosine_similarity_global, rouge_scorer_global, sent_tokenize_global,
    extract_xml_sections, safe_get_content, HUGGINGFACE_TOKEN
)

# --- Individual Reward Components (V5/V8 Style - Float16 Attempt) ---

def strict_format_adherence_v5(completions: List[Union[str, Dict, List[Dict]]], **kwargs: Any) -> List[FP_TYPE]:
    rewards = []
    cfg = kwargs.get("cfg", global_reward_config) # Allow passing config for testing
    min_tokens = cfg.get('min_format_tokens', 3)

    for i, completion_item in enumerate(completions):
        response = safe_get_content(completion_item, target_role='assistant') # Get assistant's response
        reward_val = FP_TYPE(-10.0)
        if response and isinstance(response, str):
            # Check for presence of both tags in any order, then content
            has_summary_tag_pair = bool(re.search(r"<summary>.*?</summary>", response, re.DOTALL | re.IGNORECASE))
            has_reasoning_tag_pair = bool(re.search(r"<verbatim_support>.*?</verbatim_support>", response, re.DOTALL | re.IGNORECASE))

            reasoning, summary = extract_xml_sections(response)

            if has_summary_tag_pair and has_reasoning_tag_pair:
                # Both tag pairs are present, now check content
                reasoning_ok = len(reasoning.split()) >= min_tokens
                summary_ok = len(summary.split()) >= min_tokens
                if reasoning_ok and summary_ok: reward_val = FP_TYPE(2.0)
                elif not reasoning_ok and not summary_ok: reward_val = FP_TYPE(-10.0) # Both empty
                else: reward_val = FP_TYPE(-8.0) # One is empty
            elif has_summary_tag_pair or has_reasoning_tag_pair: # Only one pair of tags
                 reward_val = FP_TYPE(-6.0)
            else: # Neither tag pair is present
                 reward_val = FP_TYPE(-10.0)
        rewards.append(reward_val)

    while len(rewards) < len(completions): rewards.append(FP_TYPE(-10.0))
    return rewards[:len(completions)]


def snippet_fidelity_v5(prompts: List[Union[str, Dict, List[Dict]]], completions: List[Union[str, Dict, List[Dict]]], **kwargs: Any) -> List[FP_TYPE]:
    rewards = []
    cfg = kwargs.get("cfg", global_reward_config)
    strict = cfg.get('strict_fidelity', True)
    min_snippet_len = cfg.get('min_snippet_len', 10)

    for i, completion_item in enumerate(completions):
        original_text = safe_get_content(prompts[i], index_in_list=-1, target_role='user') # Get last user message
        response = safe_get_content(completion_item, target_role='assistant')
        reward_val = FP_TYPE(-8.0)

        if original_text and response:
            try:
                reasoning, _ = extract_xml_sections(response)
                if not reasoning:
                    reward_val = FP_TYPE(-6.0) if re.search(r'<verbatim_support>', response, re.IGNORECASE) else FP_TYPE(-8.0)
                else:
                    potential_snippets = [s.strip().strip('"').strip("'").strip() for s in reasoning.split('\n') if s.strip() and len(s.strip()) >= min_snippet_len]
                    if not potential_snippets:
                        reward_val = FP_TYPE(-6.0)
                    else:
                        snippet_count, non_snippet_count = 0, 0
                        for snippet in potential_snippets:
                            if snippet in original_text: snippet_count += 1
                            else: non_snippet_count += 1
                        total_potential = snippet_count + non_snippet_count
                        if snippet_count == 0: reward_val = FP_TYPE(-5.0)
                        elif non_snippet_count > 0 and strict: reward_val = FP_TYPE(-5.0)
                        elif non_snippet_count == 0: reward_val = FP_TYPE(5.0)
                        else:
                            snippet_ratio = FP_TYPE(snippet_count) / FP_TYPE(total_potential + EPS)
                            reward_val = FP_TYPE(10.0) * snippet_ratio - FP_TYPE(5.0)
            except Exception:
                reward_val = FP_TYPE(-4.0)
        rewards.append(reward_val)

    while len(rewards) < len(completions): rewards.append(FP_TYPE(-8.0))
    return rewards[:len(completions)]


def snippet_relevance_v5(prompts: List[Union[str, Dict, List[Dict]]], completions: List[Union[str, Dict, List[Dict]]], **kwargs: Any) -> List[FP_TYPE]:
    if semantic_model is None or cosine_similarity_global is None or sent_tokenize_global is None:
        return [FP_TYPE(0.0)] * len(completions)
    cfg = kwargs.get("cfg", global_reward_config)
    relevance_threshold = FP_TYPE(cfg.get('relevance_threshold', 0.55))
    min_snippet_len = cfg.get('min_snippet_len', 10)
    batch_size = cfg.get('semantic_batch_size', 128)
    temp_rewards = {}
    all_reasoning_snippets, all_summary_sentences = [], []
    completion_indices, original_indices_map = [], {}
    valid_completion_count = 0

    for i, comp_item in enumerate(completions):
        response = safe_get_content(comp_item, target_role='assistant')
        temp_rewards[i] = FP_TYPE(-10.0)
        if not response: continue
        reasoning, summary = extract_xml_sections(response)
        if not reasoning or not summary: temp_rewards[i] = FP_TYPE(-4.0); continue
        reasoning_sents = [s.strip() for s in reasoning.split('\n') if s.strip() and len(s.strip()) >= min_snippet_len]
        summary_sents = [s.strip() for s in sent_tokenize_global(summary) if s.strip()]
        if not reasoning_sents or not summary_sents: temp_rewards[i] = FP_TYPE(-3.0); continue

        start_r_idx, start_s_idx = len(all_reasoning_snippets), len(all_summary_sentences)
        all_reasoning_snippets.extend(reasoning_sents); all_summary_sentences.extend(summary_sents)
        completion_indices.append({'r_slice': slice(start_r_idx, len(all_reasoning_snippets)),
                                   's_slice': slice(start_s_idx, len(all_summary_sentences)), 'orig_idx': i})
        original_indices_map[valid_completion_count] = i; valid_completion_count += 1; temp_rewards[i] = FP_TYPE(0.0)

    if valid_completion_count == 0: return [temp_rewards.get(i, FP_TYPE(-4.0)) for i in range(len(completions))]
    try:
        r_embeds = np.asarray(semantic_model.encode(all_reasoning_snippets, batch_size=batch_size, normalize_embeddings=True), dtype=FP_TYPE)
        s_embeds = np.asarray(semantic_model.encode(all_summary_sentences, batch_size=batch_size, normalize_embeddings=True), dtype=FP_TYPE)
        for info in completion_indices:
            orig_idx = info['orig_idx']
            comp_r_embeds = r_embeds[info['r_slice']]; comp_s_embeds = s_embeds[info['s_slice']]
            r_count, s_count = comp_r_embeds.shape[0], comp_s_embeds.shape[0]
            if r_count == 0 or s_count == 0: temp_rewards[orig_idx] = FP_TYPE(-3.0); continue
            sim_matrix = cosine_similarity_global(comp_r_embeds, comp_s_embeds)
            max_sim_per_snippet = np.max(sim_matrix, axis=1)
            if max_sim_per_snippet.size == 0: temp_rewards[orig_idx] = FP_TYPE(-3.0); continue
            avg_max_rel = np.mean(max_sim_per_snippet, dtype=FP_TYPE)
            irrel_ratio = np.sum(max_sim_per_snippet < relevance_threshold, dtype=FP_TYPE) / FP_TYPE(r_count + EPS)
            reward = np.clip((avg_max_rel * FP_TYPE(5.0)) - (irrel_ratio * FP_TYPE(3.0)), FP_TYPE(-3.0), FP_TYPE(5.0))
            temp_rewards[orig_idx] = reward
    except Exception:
        for idx in range(valid_completion_count): temp_rewards[original_indices_map[idx]] = FP_TYPE(-2.0)
    return [temp_rewards.get(i, FP_TYPE(-4.0)) for i in range(len(completions))]


def summary_support_semantic_v2(prompts: List[Union[str, Dict, List[Dict]]], completions: List[Union[str, Dict, List[Dict]]], **kwargs: Any) -> List[FP_TYPE]:
    if semantic_model is None: return [FP_TYPE(0.0)] * len(completions)
    cfg = kwargs.get("cfg", global_reward_config)
    min_snippet_len = cfg.get('min_snippet_len', 10)
    batch_size = cfg.get('semantic_batch_size', 128)
    s_thresh = FP_TYPE(cfg.get('semantic_support_threshold', 0.65))
    r_scale = FP_TYPE(cfg.get('semantic_reward_scale', 4.0))
    p_scale = FP_TYPE(cfg.get('semantic_penalty_scale', 4.0))
    temp_rewards = {}
    all_claims, all_snippets = [], []
    comp_indices, orig_map = [], {}
    valid_count = 0

    for i, comp_item in enumerate(completions):
        response = safe_get_content(comp_item, target_role='assistant')
        temp_rewards[i] = FP_TYPE(-8.0)
        if not response: continue
        reasoning, summary = extract_xml_sections(response)
        if not reasoning or not summary: continue
        claims_list = [s.strip() for s in sent_tokenize_global(summary) if s.strip()]
        snippets_list = [s.strip() for s in reasoning.split('\n') if s.strip() and len(s.strip()) >= min_snippet_len]
        if not claims_list or not snippets_list: temp_rewards[i] = FP_TYPE(-6.0); continue

        num_c, num_s = len(claims_list), len(snippets_list)
        c_start, s_start = len(all_claims), len(all_snippets)
        all_claims.extend(claims_list); all_snippets.extend(snippets_list)
        comp_indices.append({'num_c': num_c, 'num_s': num_s, 'c_slice': slice(c_start, len(all_claims)),
                             's_slice': slice(s_start, len(all_snippets)), 'orig_idx': i})
        orig_map[valid_count] = i; valid_count += 1; temp_rewards[i] = FP_TYPE(0.0)

    if valid_count == 0: return [temp_rewards.get(i, FP_TYPE(-8.0)) for i in range(len(completions))]
    try:
        c_embeds = np.asarray(semantic_model.encode(all_claims, batch_size=batch_size, normalize_embeddings=True), dtype=FP_TYPE)
        s_embeds = np.asarray(semantic_model.encode(all_snippets, batch_size=batch_size, normalize_embeddings=True), dtype=FP_TYPE)
        for info in comp_indices:
            orig_idx = info['orig_idx']
            num_c, num_s = info['num_c'], info['num_s']
            comp_c_embeds = c_embeds[info['c_slice']]; comp_s_embeds = s_embeds[info['s_slice']]
            if comp_c_embeds.shape[0] != num_c or comp_s_embeds.shape[0] != num_s: temp_rewards[orig_idx] = FP_TYPE(-6.0); continue
            sim_matrix = cosine_similarity_global(comp_c_embeds, comp_s_embeds)
            max_sim_per_claim = np.max(sim_matrix, axis=1)
            if max_sim_per_claim.size == 0: temp_rewards[orig_idx] = FP_TYPE(-6.0); continue
            avg_max_support = np.mean(max_sim_per_claim, dtype=FP_TYPE)
            unsupport_ratio = np.sum(max_sim_per_claim < s_thresh, dtype=FP_TYPE) / FP_TYPE(num_c + EPS)
            reward = np.clip((avg_max_support * r_scale) - (unsupport_ratio * p_scale), -p_scale, r_scale)
            temp_rewards[orig_idx] = reward
    except Exception:
        for idx in range(valid_count): temp_rewards[orig_map[idx]] = FP_TYPE(-2.0)
    return [temp_rewards.get(i, FP_TYPE(-8.0)) for i in range(len(completions))]


def summary_support_nli_v3(prompts: List[Union[str, Dict, List[Dict]]], completions: List[Union[str, Dict, List[Dict]]], **kwargs: Any) -> List[FP_TYPE]:
    if nli_model is None or nli_tokenizer is None: return [FP_TYPE(0.0)] * len(completions)
    cfg = kwargs.get("cfg", global_reward_config)
    nli_batch_size = cfg.get('nli_batch_size', 64)
    min_snippet_len = cfg.get('min_snippet_len', 10)
    r_scale = FP_TYPE(cfg.get('nli_reward_scale', 5.0))
    p_scale = FP_TYPE(cfg.get('nli_penalty_scale', 5.0))
    s_thresh = FP_TYPE(cfg.get('nli_support_threshold', 0.7))
    temp_rewards = {}
    all_nli_pairs, comp_indices = [], []
    orig_map = {}; valid_count = 0

    for i, comp_item in enumerate(completions):
        response = safe_get_content(comp_item, target_role='assistant')
        temp_rewards[i] = FP_TYPE(-8.0)
        if not response: continue
        reasoning, summary = extract_xml_sections(response)
        if not reasoning or not summary: continue
        claims = [s.strip() for s in sent_tokenize_global(summary) if s.strip()]
        snippets = [s.strip() for s in reasoning.split('\n') if s.strip() and len(s.strip()) >= min_snippet_len]
        if not claims or not snippets: temp_rewards[i] = FP_TYPE(-6.0); continue

        num_c, num_s = len(claims), len(snippets)
        pair_start = len(all_nli_pairs)
        for claim in claims:
            for snippet in snippets: all_nli_pairs.append((snippet, claim)) # premise, hypothesis
        comp_indices.append({'num_c': num_c, 'num_s': num_s, 'slice': slice(pair_start, len(all_nli_pairs)), 'orig_idx': i})
        orig_map[valid_count] = i; valid_count += 1; temp_rewards[i] = FP_TYPE(0.0)

    if valid_count == 0: return [temp_rewards.get(i, FP_TYPE(-8.0)) for i in range(len(completions))]
    try:
        all_entail_probs = []
        for j in range(0, len(all_nli_pairs), nli_batch_size):
            batch = all_nli_pairs[j:j + nli_batch_size]
            inputs = nli_tokenizer([p for p, h in batch], [h for p, h in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad(): logits = nli_model(**inputs).logits
            ent_id = nli_model.config.label2id.get("entailment", nli_model.config.label2id.get("ENTAILMENT", 2))
            probs = torch.softmax(logits.float(), dim=-1) # softmax in fp32 for stability
            all_entail_probs.extend(probs[:, ent_id].cpu().numpy().astype(FP_TYPE))
        all_entail_probs_np = np.array(all_entail_probs, dtype=FP_TYPE)

        for info in comp_indices:
            orig_idx = info['orig_idx']
            num_c, num_s = info['num_c'], info['num_s']
            comp_probs = all_entail_probs_np[info['slice']]
            if comp_probs.size != num_c * num_s: temp_rewards[orig_idx] = FP_TYPE(-6.0); continue
            prob_matrix = comp_probs.reshape((num_c, num_s))
            max_entail_per_claim = np.max(prob_matrix, axis=1)
            if max_entail_per_claim.size == 0: temp_rewards[orig_idx] = FP_TYPE(-6.0); continue
            avg_max_support = np.mean(max_entail_per_claim, dtype=FP_TYPE)
            unsupport_ratio = np.sum(max_entail_per_claim < s_thresh, dtype=FP_TYPE) / FP_TYPE(num_c + EPS)
            reward = np.clip((avg_max_support * r_scale) - (unsupport_ratio * p_scale), -p_scale, r_scale)
            temp_rewards[orig_idx] = reward
    except Exception:
        for idx in range(valid_count): temp_rewards[orig_map[idx]] = FP_TYPE(-2.0)
    return [temp_rewards.get(i, FP_TYPE(-8.0)) for i in range(len(completions))]


def reference_similarity_v5(prompts: List[Union[str, Dict, List[Dict]]],
                               completions: List[Union[str, Dict, List[Dict]]],
                               answer: List[str] | None, **kwargs: Any) -> List[FP_TYPE]:
    if semantic_model is None or rouge_scorer_global is None: return [FP_TYPE(0.0)] * len(completions)
    cfg = kwargs.get("cfg", global_reward_config)
    w_sem = FP_TYPE(cfg.get('ref_w_semantic', 0.6))
    w_rouge = FP_TYPE(cfg.get('ref_w_rouge', 0.4))
    rewards = []
    has_ans = isinstance(answer, list) and len(answer) == len(completions)

    for i, comp_item in enumerate(completions):
        response = safe_get_content(comp_item, target_role='assistant')
        ref_summary = answer[i] if has_ans and isinstance(answer[i], str) else None
        reward_val = FP_TYPE(-10.0)

        if response and ref_summary:
            _, gen_summary = extract_xml_sections(response)
            if gen_summary:
                try:
                    gen_emb = np.asarray(semantic_model.encode([gen_summary], normalize_embeddings=True), dtype=FP_TYPE)
                    ref_emb = np.asarray(semantic_model.encode([ref_summary], normalize_embeddings=True), dtype=FP_TYPE)
                    sem_sim = np.clip(cosine_similarity_global(gen_emb, ref_emb)[0][0], FP_TYPE(0.0), FP_TYPE(1.0))
                    rouge_l = np.clip(FP_TYPE(rouge_scorer_global.score(ref_summary, gen_summary)['rougeL'].fmeasure), FP_TYPE(0.0), FP_TYPE(1.0))
                    comb_score = (sem_sim * w_sem + rouge_l * w_rouge)
                    # Power in fp32 for stability
                    reward_val = np.clip(FP_TYPE(15.0) * (comb_score.astype(np.float32) ** 1.5).astype(FP_TYPE) - FP_TYPE(5.0), FP_TYPE(-5.0), FP_TYPE(10.0))
                except Exception:
                    reward_val = FP_TYPE(-2.0)
        elif response and not ref_summary: reward_val = FP_TYPE(0.0) # No reference available
        rewards.append(reward_val)

    while len(rewards) < len(completions): rewards.append(FP_TYPE(0.0))
    return rewards[:len(completions)]


def summary_conciseness_v5(prompts: List[Union[str, Dict, List[Dict]]], completions: List[Union[str, Dict, List[Dict]]], **kwargs: Any) -> List[FP_TYPE]:
    cfg = kwargs.get("cfg", global_reward_config)
    max_r = FP_TYPE(cfg.get('conciseness_max_reward', 1.5))
    sig_factor = FP_TYPE(cfg.get('conciseness_sigma_factor', 0.5))
    over_factor = FP_TYPE(cfg.get('conciseness_oversize_factor', 1.8))
    over_penalty_scale = FP_TYPE(cfg.get('conciseness_penalty_scale', 4.0))
    rewards = []

    for i, comp_item in enumerate(completions):
        orig_text = safe_get_content(prompts[i], index_in_list=-1, target_role='user')
        response = safe_get_content(comp_item, target_role='assistant')
        reward_val = FP_TYPE(-8.0)

        if orig_text and response:
            try:
                _, summary = extract_xml_sections(response)
                if not summary: reward_val = FP_TYPE(-8.0)
                else:
                    orig_tokens = len(orig_text.split())
                    sum_tokens = len(summary.split())
                    if orig_tokens == 0: reward_val = FP_TYPE(-1.0) if sum_tokens > 0 else FP_TYPE(0.0)
                    elif sum_tokens == 0: reward_val = FP_TYPE(-8.0)
                    else:
                        comp_ratio = FP_TYPE(sum_tokens) / FP_TYPE(orig_tokens + EPS)
                        ideal_r = FP_TYPE(0.20)
                        if orig_tokens > 1000: ideal_r = FP_TYPE(0.10)
                        elif orig_tokens > 500: ideal_r = FP_TYPE(0.15)
                        sigma = ideal_r * sig_factor
                        dev = comp_ratio - ideal_r
                        reward = max_r * FP_TYPE(np.exp(-0.5 * (dev.astype(np.float32) / max(sigma.astype(np.float32), EPS))**2))
                        over_thresh = ideal_r * over_factor
                        if comp_ratio > over_thresh:
                            penalty = (comp_ratio / max(over_thresh, FP_TYPE(EPS)) - FP_TYPE(1.0)) * over_penalty_scale
                            reward -= penalty
                        reward_val = np.clip(reward, FP_TYPE(-3.0), max_r)
            except Exception:
                reward_val = FP_TYPE(-2.0)
        rewards.append(reward_val)

    while len(rewards) < len(completions): rewards.append(FP_TYPE(-8.0))
    return rewards[:len(completions)]


def summary_coherence_v5(prompts: List[Union[str, Dict, List[Dict]]], completions: List[Union[str, Dict, List[Dict]]], **kwargs: Any) -> List[FP_TYPE]:
    if semantic_model is None: return [FP_TYPE(0.0)] * len(completions)
    cfg = kwargs.get("cfg", global_reward_config)
    ideal_sim = FP_TYPE(cfg.get('ideal_coherence_sim', 0.55))
    tolerance = FP_TYPE(cfg.get('coherence_tolerance', 0.15))
    batch_size = cfg.get('semantic_batch_size', 128)
    max_r = FP_TYPE(cfg.get('coherence_max_reward', 1.0))
    min_r = FP_TYPE(cfg.get('coherence_min_reward', -1.0))
    temp_rewards = {}
    sents_to_embed, comp_indices = [], []
    orig_map = {}; valid_count = 0

    for i, comp_item in enumerate(completions):
        response = safe_get_content(comp_item, target_role='assistant')
        temp_rewards[i] = FP_TYPE(-8.0)
        if not response: continue
        _, summary = extract_xml_sections(response)
        if not summary: continue
        sents = [s.strip() for s in sent_tokenize_global(summary) if s.strip()]
        num_sents = len(sents)
        if num_sents <= 1: temp_rewards[i] = FP_TYPE(0.0) if num_sents == 1 else FP_TYPE(-8.0); continue
        sents_to_embed.extend(sents)
        comp_indices.append({'count': num_sents, 'orig_idx': i})
        orig_map[valid_count] = i; valid_count += 1; temp_rewards[i] = FP_TYPE(0.0)

    if valid_count == 0: return [temp_rewards.get(i, FP_TYPE(-8.0)) for i in range(len(completions))]
    try:
        all_embeds = np.asarray(semantic_model.encode(sents_to_embed, batch_size=batch_size, normalize_embeddings=True), dtype=FP_TYPE)
        offset = 0
        for info in comp_indices:
            orig_idx = info['orig_idx']
            count = info['count']; comp_embeds = all_embeds[offset : offset + count]
            if count <= 1: temp_rewards[orig_idx] = FP_TYPE(0.0); offset += count; continue
            sims = [cosine_similarity_global(comp_embeds[j].reshape(1,-1), comp_embeds[j+1].reshape(1,-1))[0][0] for j in range(count - 1)]
            if not sims: temp_rewards[orig_idx] = FP_TYPE(0.0); offset += count; continue
            avg_sim = np.mean(np.array(sims, dtype=FP_TYPE), dtype=FP_TYPE)
            dev = avg_sim - ideal_sim
            reward_gauss = max_r * FP_TYPE(np.exp(-0.5 * (dev.astype(np.float32) / max(tolerance.astype(np.float32), EPS))**2))
            reward = np.clip(min_r + (reward_gauss / max(max_r, FP_TYPE(EPS))) * (max_r - min_r), min_r, max_r)
            temp_rewards[orig_idx] = reward
            offset += count
    except Exception:
        for idx in range(valid_count): temp_rewards[orig_map[idx]] = FP_TYPE(-2.0)
    return [temp_rewards.get(i, FP_TYPE(-8.0)) for i in range(len(completions))]


# --- Combined Reward Function (V8 - Float16 Attempt) ---
def combined_reward_function_v8(
    prompts: List[Union[str, Dict, List[Dict]]],
    completions: List[Union[str, Dict, List[Dict]]],
    answer: List[str] | None = None,
    **kwargs: Any
    ) -> List[float]: # Returns list of standard Python floats for TRL

    # Use a mutable copy of the global config for this run if needed
    current_cfg = global_reward_config.copy()
    if "cfg" in kwargs: # Allow overriding config for testing/flexibility
        current_cfg.update(kwargs["cfg"])

    if not completions: return []
    num_completions = len(completions)
    if not prompts or len(prompts) != num_completions: return [0.0] * num_completions
    if semantic_model is None : return [0.0] * num_completions # Critical dependency

    print_details = current_cfg.get('print_details', False)
    apply_raw_clip = current_cfg.get('apply_raw_reward_clip', True)
    reward_clip_min = FP_TYPE(current_cfg.get('reward_clip_min', -200.0))
    reward_clip_max = FP_TYPE(current_cfg.get('reward_clip_max', 200.0))
    use_nli_flag = current_cfg.get('use_nli_support', True)

    nli_ready = nli_model is not None and nli_tokenizer is not None
    use_nli_actual = use_nli_flag and nli_ready
    summary_support_func = summary_support_nli_v3 if use_nli_actual else summary_support_semantic_v2
    summary_support_key = 'summary_support_nli' if use_nli_actual else 'summary_support_semantic'

    if use_nli_flag and not nli_ready and not current_cfg.get('_printed_nli_fallback_warning', False):
        print("Warning [Combined V8]: NLI requested but model/tokenizer unavailable. Using semantic backup.")
        current_cfg['_printed_nli_fallback_warning'] = True # Avoid spamming

    weights = {
        'format_content': FP_TYPE(current_cfg.get('w_format', 4.0)),
        'fidelity': FP_TYPE(current_cfg.get('w_fidelity', 3.0)),
        summary_support_key: FP_TYPE(current_cfg.get('w_nli_support', 6.0) if use_nli_actual else current_cfg.get('w_semantic_support', 5.0)),
        'reasoning_relevance': FP_TYPE(current_cfg.get('w_relevance', 2.0)),
        'reference_similarity': FP_TYPE(current_cfg.get('w_reference', 5.0)),
        'conciseness': FP_TYPE(current_cfg.get('w_conciseness', 1.5)),
        'coherence': FP_TYPE(current_cfg.get('w_coherence', 1.0)),
    }
    pass_kwargs = {"cfg": current_cfg} # Pass current config to sub-functions

    all_reward_lists = {
        'format_content': strict_format_adherence_v5(completions, **pass_kwargs),
        'fidelity': snippet_fidelity_v5(prompts, completions, **pass_kwargs),
        'reasoning_relevance': snippet_relevance_v5(prompts, completions, **pass_kwargs),
        summary_support_key: summary_support_func(prompts, completions, **pass_kwargs),
        'reference_similarity': reference_similarity_v5(prompts, completions, answer, **pass_kwargs),
        'conciseness': summary_conciseness_v5(prompts, completions, **pass_kwargs),
        'coherence': summary_coherence_v5(prompts, completions, **pass_kwargs),
    }

    total_rewards_fp16 = []
    for i in range(num_completions):
        total_reward_sample = FP_TYPE(0.0)
        current_weighted_rewards_details = {} # For printing
        for key, weight_val in weights.items():
            raw_score_list = all_reward_lists.get(key)
            raw_score = FP_TYPE(0.0)
            if raw_score_list and i < len(raw_score_list):
                raw_score = raw_score_list[i] if isinstance(raw_score_list[i], FP_TYPE) else FP_TYPE(raw_score_list[i])
            weighted_score = weight_val * raw_score
            total_reward_sample += weighted_score
            current_weighted_rewards_details[key] = (raw_score, weighted_score)

        if apply_raw_clip:
            total_reward_sample = np.clip(total_reward_sample, reward_clip_min, reward_clip_max)
        total_rewards_fp16.append(total_reward_sample)

        if print_details:
            print(f"\n--- Sample {i} ---")
            prompt_content = safe_get_content(prompts[i], index_in_list=-1, target_role='user')
            response_content = safe_get_content(completions[i], target_role='assistant')
            reasoning, summary = extract_xml_sections(response_content) if response_content else ("", "")
            ref_sum = answer[i] if answer and i < len(answer) else "N/A"

            print(f"Prompt (excerpt): {prompt_content[:100] + '...' if prompt_content else 'N/A'}")
            print(f"Reference Summary: {ref_sum[:100] + '...' if ref_sum else 'N/A'}")
            print(f"Generated Summary: {summary[:100] + '...' if summary else '--- EMPTY ---'}")
            # print(f"Reasoning (excerpt): {reasoning[:100]+'...' if reasoning else '--- EMPTY ---'}")
            print("--- Weighted Reward Breakdown (float16) ---")
            sum_weighted_pre_clip_print = FP_TYPE(0.0)
            for key, (raw, weighted) in current_weighted_rewards_details.items():
                w_print = weights.get(key, FP_TYPE(0.0))
                print(f"  {key:<25}: {weighted:>9.3f}  (Raw: {raw:>7.3f}, W: {w_print:<5.1f})")
                sum_weighted_pre_clip_print += weighted
            print("------------------------------------")
            print(f"Total Score (Raw Sum, fp16): {sum_weighted_pre_clip_print:.3f}")
            if apply_raw_clip and total_reward_sample != sum_weighted_pre_clip_print:
                 print(f"Total Score (Clipped, fp16): {total_reward_sample:.3f}")
            else:
                 print(f"Total Score (Final, fp16):   {total_reward_sample:.3f}")
            print("-" * 40)

    return [float(r) for r in total_rewards_fp16] # Return standard floats