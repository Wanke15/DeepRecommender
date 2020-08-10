# DeepRecommender
Deep learning in recommendation

#### 1. Wide and deep api
- `[POST]`
```python
http://localhost:3723/wide-and-deep/adult
```

- single input：
```json
{
  "age": 28,
  "capital_gain": 0,
  "capital_loss": 0,
  "education": "Assoc-acdm",
  "education_num": 12,
  "gender": "Male",
  "hours_per_week": 40,
  "marital_status": "Married-civ-spouse",
  "native_country": "United-States",
  "occupation": "Protective-serv",
  "race": "White",
  "relationship": "Husband",
  "workclass": "Local-gov"
}
```

- single output：
```json
{
    "code": 200,
    "data": {
        "class": ">50K",
        "prob": 0.4227050840854645
    },
    "msg": "success"
}
```

- `[POST]`
```python
http://localhost:3723/wide-and-deep/adult-batch
```

- batch input：
```json
[
  {
    "age": 25,
    "capital_gain": 0,
    "capital_loss": 0,
    "education": "11th",
    "education_num": 7,
    "gender": "Male",
    "hours_per_week": 40,
    "marital_status": "Never-married",
    "native_country": "United-States",
    "occupation": "Machine-op-inspct",
    "race": "Black",
    "relationship": "Own-child",
    "workclass": "Private"
  },
  {
    "age": 28,
    "capital_gain": 0,
    "capital_loss": 0,
    "education": "Assoc-acdm",
    "education_num": 12,
    "gender": "Male",
    "hours_per_week": 40,
    "marital_status": "Married-civ-spouse",
    "native_country": "United-States",
    "occupation": "Protective-serv",
    "race": "White",
    "relationship": "Husband",
    "workclass": "Local-gov"
  }
]
```
- batch output：
```json
{
    "code": 200,
    "data": [
        {
            "class": "<=50K",
            "prob": 0.005543200299143791
        },
        {
            "class": ">50K",
            "prob": 0.4227050840854645
        }
    ],
    "msg": "success"
}
```