## Mortality prediction is one candidate mechanism for creating the value function.  The supervised_learning_dataset.py has code to compute mortality prediction from the CHOC data (need to adapt to MIMIC IV). 

The model is trained on all features aside from the actions and mortality. Mortality is transformed such that all timepoints take on the final value (0 for survivors, 1 for non-survivors).

The code uses a nested cross-validation procedure to allow for both a grid search for optimal hyperparameters, as well as the for inference using the trained model at all timesteps. 
1. First, the data is split into 5 folds (we can change this to 10 later). 
   - This is the outer split. 
   - From this, we get 5 different training-test splits
2. From each of the outer 5 training-test splits, we hold out the test fold, and create 5 new folds from the training split.
   - This is the the inner split.
   - Using this inner split, we perform a hyperparameter search using cross-validation.
   - Once the cross-validation procedure ends and optimal hyperparameters are identified, a new model is trained on the entire outer training split.
3. Using the model trained on the outer training split, predictions are then made using the held out outer test split.
   - This procedure is repeated for each of the outer training-test splits. 
    
    

