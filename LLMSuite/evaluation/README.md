# Evaluation 

**Steps**:
- Create a data file in "evaluation_data/", with entries like below (can be reconfigured)
```
{
    "sentence": sentence,
    "label": one of MAJOR_DECREASE, MINOR_DECREASE, NO_CHANGE, MINOR_INCREASE, MAJOR_INCREASE
}
```
- Set a config yaml file, with path to data file and list of models (with model parameters)
- run evaluate.py