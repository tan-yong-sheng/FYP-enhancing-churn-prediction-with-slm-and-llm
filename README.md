## FYP - Enhancing Customer Churn Prediction Using Data Mining Approaches, Small Language Model (SLMs), and Large Language Model (LLMs)


## Synthentic Data Generation

(i) Using LLM to generate synthentic feedback based on original feedback category with only 4 unique values

- Models used: `grok-3-mini-beta`
- Prompt used: 



(ii) Using LLM as judge to evaluate synthetic feedback

- Models used: `grok-3-fast-beta`
- Prompt used: 


## Starting all necessary Docker services for this project

```bash
docker compose -f docker/docker-compose.yml up -d
```


## Manage datasets to Huggingface (i.e., Upload & Download)

> **Pre-requisites**: Need to add `HF_TOKEN` environment variable to `.env` file. See more here on how to get access token for your Huggingface account: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)

- Upload local `data/` folder to my huggingface datasets repo:
  
  ```bash
  python utils/hf_data_sync.py upload --repo-id tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm --local-path data --verbose
  ```

- Download from repo to local `data/` folder:
  
  ```bash
  python utils/hf_data_sync.py download --repo-id tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm --local-path data --verbose
  ```
