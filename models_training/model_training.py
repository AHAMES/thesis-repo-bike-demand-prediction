




from hyperopt.pyll.base import scope
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
# check xgboost version
from xgboost import XGBRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        """
        Args:
            features (numpy.ndarray or torch.Tensor): Shape (num_samples, sequence_length, num_features)
            targets (numpy.ndarray or torch.Tensor): Shape (num_samples,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM layer
        out, (hn, cn) = self.lstm(x)
        # Use the last hidden state for prediction
        out = self.fc(hn[-1])
        return out

def fit_lstm(dataloader, FEATURES, batch_size, epochs):
    # Model hyperparameters
    input_dim = len(FEATURES)
    hidden_dim = 64
    num_layers = 2
    output_dim = batch_size  # Regression output

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = epochs
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_targets in dataloader:
            # Reset gradients
            optimizer.zero_grad()
            
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            # Forward pass
            outputs = model(batch_features)
            
            smallest = min(outputs.shape[0], batch_targets.shape[0])
            outputs = outputs[0:smallest]
            batch_targets = batch_targets[0:smallest]
            loss = criterion(outputs.squeeze(), batch_targets)

            # Backward pass
            loss.backward()
            optimizer.step()
            # Accumulate loss
            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

    return model

def test_lstm(model, dataloader, FEATURES, batch_size):
  model = model.to(device)

  output_df = pd.DataFrame()
  for batch_features, batch_targets in dataloader:
      # Reset gradients
      
      batch_features = batch_features.to(device)
      batch_targets = batch_targets.to(device)
      # Forward pass
      outputs = model(batch_features)
      
      smallest = min(outputs.shape[0], batch_targets.shape[0])
      outputs = outputs[0:smallest]
      batch_targets = batch_targets[0:smallest]

      y_pred = target_scaler.inverse_transform([outputs.detach().numpy()])[0]
      current_df_out = pd.DataFrame(batch_features, columns = FEATURES)
      current_df_out['actual_demand'] = batch_targets.detach().numpy()
      current_df_out['pred'] = y_pred

      output_df = pd.concat([output_df, current_df_out])

  mse_score = mean_squared_error(output_df['actual_demand'], output_df['pred'])
  rmse_score = np.sqrt(mean_squared_error(output_df['actual_demand'], output_df['pred']))
  mae_score = mean_absolute_error(output_df['actual_demand'], output_df['pred'])
  output_df['start_station_cluster'] = output_df['start_station_cluster'].astype(int)
  mape_score = mean_absolute_percentage_error(output_df['actual_demand']+1, output_df['pred']+1)
  print(mse_score)
  print(rmse_score)
  print(mae_score)
  print(mape_score)

  return output_df

def tuning_xgb(X_train, y_train, X_test, y_test):
    def objective(params):
        reg = XGBRegressor(**params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        RMSE = math.sqrt(mse)
        return {"loss": RMSE, "status": STATUS_OK}

    space = {
        "n_estimators": scope.to_int(hp.quniform("n_estimators", 20, 250, 5)),
        'max_depth': scope.to_int(hp.quniform("max_depth", 3, 14, 2)),
        'learning_rate': hp.loguniform('learning_rate', 0.01, 0.3),
        
        #'gamma': hp.uniform ('gamma', 0,9),
        #'reg_alpha' : hp.quniform('reg_alpha', 0,180, 5),
        #'reg_lambda' : hp.uniform('reg_lambda', 0, 50),
        #'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1.0),
        #'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        #'seed': 42,
        #"subsample": hp.uniform("subsample", 0.1, 1.0),
        #"scale_pos_weight": hp.uniform("scale_pos_weight", 0.1, 10),
        #"max_delta_step": hp.quniform("max_delta_step", 0, 10, 1),
    }
    print("hyperopt")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=300,
        trials=trials,
    )

    best_params = {
    "n_estimators": int(best["n_estimators"]),
    "max_depth": int(best["max_depth"]),
    "learning_rate": best["learning_rate"],
    #"gamma": best["gamma"],
    #"min_child_weight": int(best["min_child_weight"]),
    #"subsample": best["subsample"],
    #"colsample_bytree": best["colsample_bytree"],
    #"reg_alpha": best["reg_alpha"],
    #"reg_lambda": best["reg_lambda"],
    #"scale_pos_weight": best["scale_pos_weight"],
    #"max_delta_step": int(best["max_delta_step"]),
    }
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    return model

def tuning_xgb_2(X_train, y_train):
    
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 2000, 5),
            'max_depth': trial.suggest_int('max_depth', 3, 18),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            
            #'max_delta_step' : trial.suggest_float('max_delta_step', 0, 10),
            #'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0.1, 10),
            #'reg_lambda' : trial.suggest_int('reg_lambda', 0, 50),
            #'reg_alpha': trial.suggest_int('reg_alpha', 0,180, 5),
            #
            #'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            #'gamma': trial.suggest_float('gamma', 0, 9),
            #'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
        }
        
        xgb = XGBRegressor(**param)
        
        score = cross_val_score(xgb, X_train, y_train, scoring='neg_mean_squared_error', cv=3).mean()
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300, n_jobs=-1)
    pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
    print("Best parameters:", study.best_params_)
    best_params = grid_search.best_params_  # or random_search.best_params_ / study.best_params_
    xgb = XGBRegressor(**best_params)
    xgb.fit(X_train, y_train, callbacks=[pruning_callback])
    
    return xgb

def train_model(df_train, df_test, FEATURES, model='xgb', epochs=50, batch_size=32, tuning=False):
    TARGET = 'demand'
    TARGET_TRAIN = 'demand_target'
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET_TRAIN]

    X_test = df_test[FEATURES]
    y_test = df_test[TARGET]
    
    overall_zero_scores = pd.DataFrame()
    non_zero_scores = pd.DataFrame()
    zero_scores = pd.DataFrame()
    preds_list = []
    for state in [10, 20, 30, 42, 50, 60, 70, 80, 90, 0]:
        if model=='rf':
            reg = RandomForestRegressor(random_state=state)
            reg.fit(X_train, y_train)
    
        if model=='xgb':
            if tuning:
               reg = tuning_xgb(X_train, y_train, X_test, y_test)
            else:
               if state==0:
                   state = None
               reg = XGBRegressor(random_state=state)
               reg.fit(X_train, y_train)
    
        if model=='lgbm':
            # Define the parameter grid
            if tuning:
                param_grid = {
                    'learning_rate': [0.01, 0.3, 0.05, 0.1, 0.2],
                    'num_leaves': [10, 20, 31, 40, 50, 60],
                    'max_depth': [-1, 10, 20, 30, 50],
                    'feature_fraction': [0.6, 0.8, 1.0],
                    'bagging_fraction': [0.6, 0.8, 1.0],
                    'n_estimators': [20, 40, 60, 80, 100, 120, 140]
                }
        
                # Initialize the LightGBM model
                model = lgb.LGBMRegressor(objective='regression', metric='rmse')
        
                # Perform grid search
                grid_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=3,
                    verbose=1,
                    n_iter=10,
                )
                grid_search.fit(X_train, y_train)
        
                # Best parameters
                #print("Best parameters:", grid_search.best_params_)
                reg = LGBMRegressor(**grid_search.best_params_)
                reg.fit(X_train, y_train)
            else:
                reg = LGBMRegressor(random_state=state)
                reg.fit(X_train, y_train)
                
    
        if model=='lstm':
    
            dataset = TimeSeriesDataset(X_train.to_numpy(), y_train.to_numpy())
            test_dataset = TimeSeriesDataset(X_test.to_numpy(), y_test.to_numpy())
    
            # Create DataLoader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            reg = fit_lstm(dataloader, FEATURES, batch_size, epochs)
            
            output_df = test_lstm(reg, test_dataloader, FEATURES, batch_size)
                
            return reg, output_df

        y_pred = target_scaler.inverse_transform([reg.predict(X_test)])
        preds_out = X_test.copy()
        preds_out['actual_demand'] = y_test
        preds_out['pred'] = y_pred[0]
        
        if model != 'rf':
            preds_list.append(preds_out)

        non_zero = preds_out.query('actual_demand != 0')
        zeros = preds_out.query('actual_demand == 0')
        
        mse_score = mean_squared_error(y_test, y_pred[0])
        rmse_score = np.sqrt(mse_score)
        mae_score = mean_absolute_error(y_test, y_pred[0])
        mape_score = mean_absolute_percentage_error(y_test+1, y_pred[0]+1)
        overall_zero_scores = pd.concat([overall_zero_scores, pd.DataFrame({'mse': [mse_score], 'rmse': [rmse_score], 'mae': [mae_score], 'mape': [mape_score]})])
        
        mse_score = mean_squared_error(non_zero['actual_demand'], non_zero['pred'])
        rmse_score = np.sqrt(mse_score)
        mae_score = mean_absolute_error(non_zero['actual_demand'], non_zero['pred'])
        mape_score = mean_absolute_percentage_error(non_zero['actual_demand']+1, non_zero['pred']+1)
        non_zero_scores = pd.concat([non_zero_scores, pd.DataFrame({'mse': [mse_score], 'rmse': [rmse_score], 'mae': [mae_score], 'mape': [mape_score]})])
        
        
        mse_score = mean_squared_error(zeros['actual_demand'], zeros['pred'])
        rmse_score = np.sqrt(mse_score)
        mae_score = mean_absolute_error(zeros['actual_demand'], zeros['pred'])
        mape_score = mean_absolute_percentage_error(zeros['actual_demand']+1, zeros['pred']+1)
        zero_scores = pd.concat([zero_scores, pd.DataFrame({'mse': [mse_score], 'rmse': [rmse_score], 'mae': [mae_score], 'mape': [mape_score]})])

    print("overall")
    print("MSE:", overall_zero_scores['mse'].mean())
    print("RMSE:", overall_zero_scores['rmse'].mean())
    print("MAE:", overall_zero_scores['mae'].mean())
    print("MAPE:", overall_zero_scores['mape'].mean())
    print("### Standard Deviation")
    print("MSE:", overall_zero_scores['mse'].std())
    print("RMSE:", overall_zero_scores['rmse'].std())
    print("MAE:", overall_zero_scores['mae'].std())
    print("MAPE:", overall_zero_scores['mape'].std())
    print("### Variance")
    print("MSE:", overall_zero_scores['mse'].var())
    print("RMSE:", overall_zero_scores['rmse'].var())
    print("MAE:", overall_zero_scores['mae'].var())
    print("MAPE:", overall_zero_scores['mape'].var())
    
    print()
    print("Non-zero")
    print("MSE:", non_zero_scores['mse'].mean())
    print("RMSE:", non_zero_scores['rmse'].mean())
    print("MAE:", non_zero_scores['mae'].mean())
    print("MAPE:", non_zero_scores['mape'].mean())
    print("### Standard Deviation")
    print("MSE:", non_zero_scores['mse'].std())
    print("RMSE:", non_zero_scores['rmse'].std())
    print("MAE:", non_zero_scores['mae'].std())
    print("MAPE:", non_zero_scores['mape'].std())
    print("### Variance")
    print("MSE:", non_zero_scores['mse'].var())
    print("RMSE:", non_zero_scores['rmse'].var())
    print("MAE:", non_zero_scores['mae'].var())
    print("MAPE:", non_zero_scores['mape'].var())
    
    print()
    print("#####")
    print("Zeros")
    print("MSE:", zero_scores['mse'].mean())
    print("RMSE:", zero_scores['rmse'].mean())
    print("MAE:", zero_scores['mae'].mean())
    print("MAPE:", zero_scores['mape'].mean())
    print("### Standard Deviation")
    print("MSE:", zero_scores['mse'].std())
    print("RMSE:", zero_scores['rmse'].std())
    print("MAE:", zero_scores['mae'].std())
    print("MAPE:", zero_scores['mape'].std())
    print("### Variance")
    print("MSE:", zero_scores['mse'].var())
    print("RMSE:", zero_scores['rmse'].var())
    print("MAE:", zero_scores['mae'].var())
    print("MAPE:", zero_scores['mape'].var())

    return reg, preds_out, preds_list

def train_model_unscaled(df_train, df_test, FEATURES, model='xgb'):
    TARGET = 'demand'
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]

    X_test = df_test[FEATURES]
    y_test = df_test[TARGET]

    if model=='rf':
        reg = RandomForestRegressor(random_state=42)
        reg.fit(X_train, y_train)

    if model=='xgb':
        reg = tuning_xgb(X_train, y_train, X_test, y_test)

    y_pred = reg.predict(X_test)
    preds_out = X_test.copy()
    preds_out['actual_demand'] = y_test
    preds_out['pred'] = y_pred

    mse_score = mean_squared_error(y_test, y_pred)
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))


    mape_score = mean_absolute_percentage_error(y_test+1, y_pred+1)
    print(mse_score)
    print(rmse_score)
    print(mape_score)

    return reg, preds_out

