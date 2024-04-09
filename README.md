# Where to Move Tells Where to Live: An In-Context Homestay Market Forecasting Framework with Human Mobility Embedding
This is an implementation of RICH: [Where to Move Tells Where to Live: An In-Context Homestay
Market Forecasting Framework with Human Mobility Embedding].
## Environment
- Python 3.9.13
- PyTorch 2.1.1+cu121
- NumPy 1.26.2
- scipy 1.7.3
- sklearn
## Dataset
Chicago data: ch_data

NewYork City data: ny_data
## Train command
    first unzip the dataset files in /ny_bnb/final_data
    # Train with model
    python train_session.py --{multiple settings}

## Table 2 Results

```
# NewYork dataset
run NewYork_train.sh
# Chicago dataset
run Chicago_train.sh
```
