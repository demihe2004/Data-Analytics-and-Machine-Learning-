# 1. Initiate a random forest 
``` python
rf = RandomForestRegressor(n_estimators = 250,
max_depth = 10,
random_state = 42)
```

# 2. Initiate a XGBoost 
``` python
xgb_regressor = xgb.XGBRegressor(objective = "reg:linear", # Specify the learning task
                                n_estimators = 100, # Number of trees in random forest to fit
                                reg_lambda = 1, # L2 regularization term
                                gamma = 0, # Minimum loss reduction
                                max_depth = 6, # Maximum tree depth
                                learning_rate = 0.1, # Learning rate, eta
                                verbosity = 0 # Ignore warnings
                                )
```

# 3. Split training and test data
```python
training_x, test_x, training_y, test_y = train_test_split(X, y,
test_size=0.2,
random_state=12345)
```

# 4. Random forest prediction
```python
rf.fit(training_x, training_y)
rf_prediction = rf.predict(test_x)
```
- MSE and pseudo R2 
```python
mse_rf_boston = mse(rf_prediction, test_y)
mse_base_boston = np.mean(np.square((test_y - np.mean(test_y))))
PseudoR2_rf = 1-mse_rf_boston / mse_base_boston
```

# 5. XGBoost
- cross validation:
```python
xgb_parm = xgb_regressor.get_xgb_params()
xgb_train = xgb.DMatrix(training_x, label = training_y)
xgb_cvresult = xgb.cv(params = xgb_parm,
                        dtrain = xgb_train,
                        num_boost_round = 200,
                        metrics = "rmse",
                        nfold = 10,
                        stratified = False,
                        seed=12345678)
```
- Train model 
```python
xgb_regressor.set_params(n_estimators = xgb_cvresult.shape[0])
xgb_regressor.fit(training_x, training_y)
# Test the model
xgb_prediction_boston = xgb_regressor.predict(test_x, ntree_limit = xgb_cvresult.shape[0])
```
- MSE and pseudo R2
```python
# Compute the average of squared errors
mse_xgb_boston = mse(xgb_prediction_boston, test_y)
# PseudoR2 for XGBoost
PseudoR2_xgb = 1 - mse_xgb_boston / mse_base_boston
```

# 6. Elastic net
- cross validation:
```python
# Cross‑validation to find best alpha
elastic_cv = ElasticNetCV(cv = 4, l1_ratio = 0.5)
elastic_cv.fit(training_x, training_y)
# Get the best alpha
best_alpha = elastic_cv.alpha_
```
- Train model 
```python
elastic = ElasticNet(alpha = best_alpha, l1_ratio = 0.5)
elastic.fit(training_x, training_y)
enet_prediction = elastic.predict(test_x)
```
- MSE and pseudo R2
```python
# Compute MSE
mse_rf_enet = mse(enet_prediction, test_y)
# PseudoR2 for random forest
PseudoR2_enet = 1-mse_rf_enet / mse_base_boston
```



