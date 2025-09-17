# FedGNN_advanced/inference.py
"""
Inference wrapper used by demo. Provides predict(question, context=None).

Behavior:
1) If a local checkpoint exists under checkpoints/ (best_model.pt etc.), try to load it.
   - Best-effort: if it's a HF-style model dir, load via AutoModelForQuestionAnswering.
   - If it's state_dict, try to map into a distilbert-small QA model for demo.
2) Otherwise, fall back to HuggingFace pipeline('question-answering', model='distilbert-base-uncased-distilled-squad').
"""

import os

_cached = {
    "local": None,
    "hf": None
}

def _load_local():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    except Exception:
        return None
    paths = ['checkpoints/qa_model.pt', 'checkpoints/best_model.pt', 'checkpoints/model.pt']
    for p in paths:
        if os.path.exists(p):
            try:
                # If folder - try to load HF style
                if os.path.isdir(p):
                    tok = AutoTokenizer.from_pretrained(p)
                    model = AutoModelForQuestionAnswering.from_pretrained(p)
                    return (tok, model)
                else:
                    sd = torch.load(p, map_location='cpu')
                    # try to load into distilbert-base model
                    base = 'distilbert-base-uncased'
                    tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
                    model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')
                    try:
                        # if state-dict contains 'model_state_dict', unwrap
                        if isinstance(sd, dict) and 'model_state_dict' in sd:
                            model.load_state_dict(sd['model_state_dict'], strict=False)
                        elif isinstance(sd, dict):
                            model.load_state_dict(sd, strict=False)
                    except Exception:
                        pass
                    return (tok, model)
            except Exception:
                continue
    return None

def predict(question: str, context: str = None):
    # try local model first
    try:
        if _cached['local'] is None:
            _cached['local'] = _load_local()
        if _cached['local'] is not None:
            tok, model = _cached['local']
            import torch
            inputs = tok(question, context or "", return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            start = outputs.start_logits.argmax().item()
            end = outputs.end_logits.argmax().item()
            tokens = inputs['input_ids'][0][start:end+1]
            answer = tok.decode(tokens, skip_special_tokens=True)
            return {'answer': answer, 'source': 'local_model'}
    except Exception:
        pass

    # fallback to HF pipeline
    try:
        if _cached['hf'] is None:
            from transformers import pipeline
            _cached['hf'] = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='distilbert-base-uncased')
        pipe = _cached['hf']
        ctx = context
        if ctx is None:
            # try to build context from data/client_1/docs.json if exists
            p = 'data/client_1/docs.json'
            if os.path.exists(p):
                try:
                    import json
                    docs = json.load(open(p, 'r', encoding='utf-8'))
                    ctx = ' '.join([d.get('text','') for d in docs[:5]])
                except Exception:
                    ctx = ""
            else:
                ctx = ""
        res = pipe(question=question, context=ctx)
        return {'answer': res.get('answer', ''), 'score': float(res.get('score', 0.0)), 'source': 'hf_pipeline'}
    except Exception as e:
        return {'error': 'inference not available: ' + str(e)}
