## FYP - Enhancing Customer Churn Prediction Using Data Mining Approaches, Small Language Model (SLMs), and Large Language Model (LLMs)


### Synthentic Data Generation

(i) Using LLM to generate synthentic feedback based on original feedback category with only 4 unique values

(ii) Using LLM as judge to evaluate synthetic feedback


## Starting Docker services

(i) CPU

```bash
docker compose -f docker/docker-compose.cpu.yml up -d
```

(ii) Nvidia GPU

```bash
docker compose -f docker/docker-compose.cuda.yml up -d
```

