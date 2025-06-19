## FYP - Enhancing Customer Churn Prediction Using Data Mining Approaches, Small Language Model (SLMs), and Large Language Model (LLMs)

Datasets for this project is stored at HuggingFace: [https://huggingface.co/datasets/tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm/tree/main](https://huggingface.co/datasets/tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm/tree/main)



## Reproducibility of this Project
### Starting all necessary Docker services for this project

```bash
docker compose -f docker/docker-compose.yml up -d
```

### Synthentic Data Generation

(i) Using LLM to generate synthentic feedback based on original feedback category with only 4 unique values



- Models used: `grok-3-mini-beta`
- Prompt used: Check [this notebook](notebook/text-representation/utils/prepare_llm_prompt.py)



(ii) Using LLM as judge to evaluate synthetic feedback

- Models used: `grok-3-fast-beta`
- Prompt used: 


### Embedding

- Models used: `text-embedding-004` from gemini

----

Reminder: refer to `reference/` folder for more references you've collected to read...


### LLM for explainability

Reminder: refer to `reference/` folder for more references you've collected to read... (note: `pycaret` conda environment)