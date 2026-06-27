# Interview Notes

## Why LangGraph Is Used

LangGraph is useful here because model evaluation is a multi-step process with explicit state. Instead of one long script, the project breaks the run into nodes: config loading, dataset preparation, training, evaluation, failure mining, failure analysis, and report generation.

That structure makes the workflow easier to inspect, test, and extend. A later version could add human approval, retries, branching, or background execution without rewriting the model code.

## How Workflow State Works

Each node receives a shared state dictionary and returns an updated state. The state stores the config, run id, output directory, dataset, model, metrics, predictions, failure cases, and generated report path.

The state is also useful for debugging because each node appends to `completed_nodes`.

## How The LSTM NER Model Works

The NER pipeline reads data where each line has one token and one tag. Blank lines separate sentences.

The model uses:

- `nn.Embedding` for token embeddings
- bidirectional `nn.LSTM` for contextual token representations
- linear classifier for token-level tag logits

The loss ignores padded positions through a boolean mask, so padding does not affect optimization.

## How The Transformer Decoder Works

The Transformer Decoder is implemented manually with PyTorch tensor operations.

It includes:

- token embeddings
- sinusoidal positional encoding
- scaled dot-product attention
- multi-head self-attention
- feed-forward network
- residual connections
- layer normalization
- output projection to vocabulary logits

It predicts the next token at each sequence position.

## What Causal Masking Is

Causal masking prevents a token from attending to future tokens. During next-token prediction, position `t` can only attend to positions `0..t`.

The attention score matrix is masked above the diagonal with `-inf`, so softmax gives future positions zero probability.

This is what makes the Transformer decoder autoregressive.

## How Failure Mining Works

For NER runs, the evaluator writes predicted tags and gold tags to `predictions.jsonl`.

The failure miner compares each token's predicted tag with the gold tag. Incorrect examples are written to `failure_cases.jsonl`.

For language modeling runs, failure mining is currently empty because there is no token-level gold tag comparison. The dashboard focuses on loss, perplexity, and generated samples.

## How The LLM Analyzer Works

The analyzer is designed as a wrapper. If no external API key is available, it uses a deterministic fallback that summarizes error patterns from the mined failure cases.

That makes the project runnable locally without external services while leaving room for a real LLM-backed analysis later.

## How The Frontend Helps Debugging

The frontend makes the system easier to inspect:

- list recent runs
- compare NER and Transformer workflows
- inspect metrics and loss history
- view predictions
- review failure cases
- read generated reports
- trigger new workflows

This turns the project from scripts into a small model evaluation product.

## What I Would Improve Next

- Add train/validation/test splits.
- Add per-tag precision, recall, and confusion matrices for NER.
- Add richer language-modeling evaluation beyond tiny sample perplexity.
- Add background jobs for workflow execution.
- Add run cancellation and retry support.
- Add human review annotations for failure cases.
- Add model comparison views in the dashboard.
