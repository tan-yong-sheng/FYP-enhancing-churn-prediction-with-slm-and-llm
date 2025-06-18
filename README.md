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
- Prompt used: 



(ii) Using LLM as judge to evaluate synthetic feedback

- Models used: `grok-3-fast-beta`
- Prompt used: 


### Embedding

...



## Manage data versioning with dvc and s3-compatible cloud (such as Cloudflare R2)

```bash
dvc remote add -d myremote s3://fyp-churn
dvc remote modify --local myremote access_key_id 'mysecret'
dvc remote modify --local myremote secret_access_key 'mysecret'
dvc remote modify myremote endpointurl 'myendpointurl'
dvc remote modify myremote region 'auto'
```

```bash
dvc add data
dvc push
```



## Reference

- https://github.com/d0r1h/Churn-Analysis

- Theoritical base: https://spotintelligence.com/2023/06/13/combining-numerical-text-features-python/

- https://medium.com/@brijesh_soni/stacking-to-improve-model-performance-a-comprehensive-guide-on-ensemble-learning-in-python-9ed53c93ce28

- https://towardsdatascience.com/customer-churn-prediction-with-text-and-interpretability-bd3d57af34b1/
- Actionable Recommendations with LLM  - Churn Mastery with AI: LLM & PyCaret in Action 🚀 https://www.kaggle.com/code/memocan/churn-mastery-with-ai-llm-pycaret-in-action/code


- https://freedium.cfd/https://medium.com/data-science/the-stacking-ensemble-method-984f5134463a

```
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Suppose:
# text_model = fine-tuned BERT
# table_model = RandomForestClassifier()

estimators = [
    ('text', text_model),
    ('table', table_model)
]
meta = LogisticRegression()

stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    cv=5,
    passthrough=False,         # If you want to include original structured features, set True
    stack_method='predict_proba'
)
stack_clf.fit(X_train, y_train)  # X_train can combine both text and structured inputs if passthrough=True
```