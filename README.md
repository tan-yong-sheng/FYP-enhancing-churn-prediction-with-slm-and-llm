## FYP - Enhancing Customer Churn Prediction Using Data Mining Approaches, Small Language Model (SLMs), and Large Language Model (LLMs)

Datasets for this project is stored at HuggingFace: [https://huggingface.co/datasets/tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm/tree/main](https://huggingface.co/datasets/tan-yong-sheng/FYP-enhancing-churn-prediction-with-slm-and-llm/tree/main)


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

## Manage data versioning with dvc and s3-compatible cloud (such as Cloudflare R2)

```bash
dvc remote add -d myremote s3://fyp-churn
dvc remote modify --local myremote access_key_id 'mysecret'
dvc remote modify --local myremote secret_access_key 'mysecret'
dvc remote modify myremote endpointurl 'myendpointurl'
dvc remote modify myremote region 'auto'
```

```bash
dvc add data/csv
dvc push
```

