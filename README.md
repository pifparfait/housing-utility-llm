# Housing Utility LLM

This project combines **Large Language Models (LLMs)** with a housing utility framework to:
- **Buyers**: rank rental properties based on personal preferences, budget, and textual descriptions.
- **Sellers**: analyze market conditions and generate useful metrics to adjust supply strategies.

---

## Repository Structure

housing-utility-llm/
├── buyer/
│ ├── housing_utility_llm_demo.py # Main buyer script (ranking engine)
│ ├── ranked_results_with_llm.csv # Full ranked results
│ └── ranked_top10_with_llm.csv # Top-10 ranked results
│
├── seller/
│ ├── housing_seller_advisor.py # Main seller script
│ ├── seller_market_metrics.csv # Raw metrics
│ └── seller_market_metrics_pretty.csv # Formatted metrics
│
├── data/ # Datasets
│ ├── rental-data-united-states_2_2_0.csv
│ ├── rental-data-united-states_2_5_0.csv
│ ├── rental-data-united-states_4_0_0.csv
│ └── rental-data-united-states_8_2_0.csv
│
├── .gitignore
└── README.md

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Buyer (Usage)
python buyer/housing_utility_llm_demo.py \
  --csv data/rental-data-united-states_8_2_0.csv \
  --salary 6000 \
  --buyer_text "Looking for at least 2 bedrooms, quiet street, good transport access" \
  --city "Boston" \
  --topk 10 \
  --max_rows 30 \
  --verbosity debug \
  --llm_timeout 20 \
  --llm_num_predict 150 \
  --llm_retries 2

## Seller (Usage)
python housing_seller_advisor.py \
  --runs_glob "ranked_results_with_llm.csv" \
  --k 10 --min_buyers 1 --verbosity info


python housing_seller_advisor.py \
  --runs_glob "ranked_results_with_llm.csv" \
  --k 10 --min_buyers 1 --gen_copy auto \
  --ollama_temperature 0.2 --ollama_num_predict 350 --ollama_timeout 20


##Configuration
Set environment variables for Ollama:
export OLLAMA_URL=http://localhost:11434/api/generate
export OLLAMA_MODEL=llama3.1
Key script parameters:
--topk: number of ranked listings to output.
--max_rows: maximum number of rows to process from the dataset.
--verbosity: logging level (debug | info | quiet).

##Testing
If you add unit tests:
pytest -q

##License
MIT License © 2025