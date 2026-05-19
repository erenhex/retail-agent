## Data pipelines

Run all commands from the **`retail-bench/`** repo root.

**Host deps** (index + synthesize):

```bash
pip install -e ".[datapipeline]"
```

- **Download** `documents.jsonl` into `resources/`:

  ```bash
  mkdir -p resources
  hf sync hf://buckets/erenhex/retail-agent ./resources
  ```

  If you already have `Datapipeline/resources/documents.jsonl`:

  ```bash
  mkdir -p resources
  mv Datapipeline/resources/documents.jsonl resources/
  ```

- **Search index** (writes `indexes/` at repo root):

  ```bash
  ./build_index.sh
  ```

- **Synthesized suites** (writes under `data/suites/`). Loads **`retail-bench/.env`** automatically (`OPENAI_API_KEY`, `OPENAI_BASE_URL`; falls back `OPENROUTER_API_KEY` → `OPENAI_API_KEY`):

  ```bash
  # One task per run
  python -m Datapipeline.src.suitegen.run_synthesize Datapipeline/config/synthesize/product.json
  python -m Datapipeline.src.suitegen.run_synthesize Datapipeline/config/synthesize/shop.json
  python -m Datapipeline.src.suitegen.run_synthesize Datapipeline/config/synthesize/voucher.json

  # All three tasks → one JSONL (LLM calls for product, shop, voucher in sequence)
  python -m Datapipeline.src.suitegen.run_synthesize Datapipeline/config/synthesize/combined.json

  # Concatenate existing per-task JSONL files (no LLM)
  python -m Datapipeline.src.suitegen.run_synthesize Datapipeline/config/synthesize/merge.json
  ```
