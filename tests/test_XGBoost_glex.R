# ===============================================
# Regional Surrogate Modeling on AlphaPEM Polarization Curves (with glex)
# We split the polarization curve into 3 regions:
# Activation (0 <= ifc < 0.4), Ohmic (0.4 <= ifc < 1.6), and Mass Transport (ifc >= 1.6)
# For each region, we compute the AUC and train an XGBoost to predict it from the 13 input parameters.
# Then, we run glex for global explanation and plot variable importance.
# ===============================================

# ------------------------
# 0. Load Packages
# ------------------------
library(data.table)
library(dplyr)
library(xgboost)
library(caret)
library(ModelMetrics)
library(yardstick)
library(ggplot2)
library(tidyr)
library(zoo)
library(glex)
library(patchwork)

theme_set(theme_glex())

# ------------------------
# 1. Load and Filter Wide Format Data
# ------------------------
wide_data <- fread("../sampling_test/validated_final_df.csv")

ucell_cols <- grep("^Ucell_", names(wide_data), value = TRUE)
ifc_cols   <- gsub("Ucell_", "ifc_", ucell_cols)

param_cols <- c("Tfc", "Pa_des", "Sc", "Phi_c_des", "epsilon_gdl", "tau",
                "epsilon_mc", "epsilon_c", "e", "Re", "i0_c_ref",
                "kappa_co", "kappa_c")

valid_data <- wide_data %>%
  filter(classification %in% c("valid", "invalid")) %>%
  select(all_of(c(param_cols, ifc_cols, ucell_cols)))

# ------------------------
# 2. Define Utility to Compute Targets per Region
# ------------------------
region_bounds <- c(0, 0.4, 1.6, Inf)
region_names <- c("activation", "ohmic", "mass_transport")

compute_region_target <- function(df_row, region = "activation", method = "auc") {
  region_idx <- match(region, region_names)
  if (is.na(region_idx)) stop("Invalid region name")
  
  ifcs <- as.numeric(df_row[ifc_cols])
  volts <- as.numeric(df_row[ucell_cols])
  
  lower <- region_bounds[region_idx]
  upper <- region_bounds[region_idx + 1]
  
  mask <- ifcs >= lower & ifcs < upper
  if (sum(mask) < 2) return(NA)
  
  if (method == "auc") {
    return(sum(diff(ifcs[mask]) * rollmean(volts[mask], 2)))
  } else if (method == "sum") {
    return(sum(volts[mask]))
  } else {
    stop("Invalid method. Use 'auc' or 'sum'.")
  }
}

# ------------------------
# 3. Compute Targets for Each Region
# ------------------------
for (rn in region_names) {
  valid_data[[paste0("target_", rn)]] <- apply(valid_data, 1, compute_region_target, region = rn, method = "auc")
}

# ------------------------
# 4. Function to Train and Evaluate Model for a Region
# ------------------------
train_model_region <- function(region_name) {
  cat("
[INFO] Training model for region:", region_name, "
")
  df <- valid_data %>% select(all_of(param_cols), !!sym(paste0("target_", region_name))) %>%
    rename(target = paste0("target_", region_name)) %>%
    filter(!is.na(target))
  
  X <- as.matrix(df %>% dplyr::select(all_of(param_cols)))
  y <- df$target
  
  params <- list(
    objective = "reg:squarederror",
    learning_rate = 0.05,
    max_depth = 8,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  set.seed(42)
  folds <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)
  metrics <- data.frame(region = region_name, MAE_train = NA, R2_train = NA, MAE_test = NA, R2_test = NA)
  mae_train <- r2_train <- mae_test <- r2_test <- numeric(length(folds))
  
  for (i in seq_along(folds)) {
    cat(paste0("[INFO] Fold ", i, " of ", length(folds), "...
"))
    test_idx <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)
    
    dtrain <- xgb.DMatrix(X[train_idx, ], label = y[train_idx])
    dtest <- xgb.DMatrix(X[test_idx, ], label = y[test_idx])
    
    model <- xgboost(data = dtrain, params = params, nrounds = 500, verbose = 0)
    
    y_train_pred <- predict(model, dtrain)
    y_test_pred <- predict(model, dtest)
    
    mae_train[i] <- ModelMetrics::mae(y[train_idx], y_train_pred)
    r2_train[i] <- yardstick::rsq_vec(truth = y[train_idx], estimate = y_train_pred)
    mae_test[i] <- ModelMetrics::mae(y[test_idx], y_test_pred)
    r2_test[i] <- yardstick::rsq_vec(truth = y[test_idx], estimate = y_test_pred)
  }
  
  metrics$MAE_train <- mean(mae_train)
  metrics$R2_train <- mean(r2_train)
  metrics$MAE_test <- mean(mae_test)
  metrics$R2_test <- mean(r2_test)
  
  cat("[INFO] Training final model on full data...
")
  final_model <- xgboost(data = xgb.DMatrix(X, label = y), params = params, nrounds = 500, verbose = 0)
  
  cat("[INFO] Finished training and evaluation. Saved metrics and model.")
  return(list(model = final_model, metrics = metrics))
}

# ------------------------
# 5. Function to Explain Model with glex
# ------------------------
explain_model_glex <- function(region_name, model_obj) {
  cat("
[INFO] Running glex explanation for region:", region_name, "
")
  df <- valid_data %>% select(all_of(param_cols), !!sym(paste0("target_", region_name))) %>%
    rename(target = paste0("target_", region_name)) %>%
    filter(!is.na(target))
  
  X <- df %>% select(all_of(param_cols)) %>% as.matrix()
  
  cat("[INFO] Running glex()...
")
  glex_obj <- glex(model_obj, X)
  cat("[INFO] Computing variable importance with glex_vi()...
")
  vi <- glex_vi(glex_obj)
  
  p_imp <- autoplot(vi, threshold = 0.05) + labs(title = paste("Importance (", region_name, ")"))
  p_deg <- autoplot(vi, by_degree = TRUE) + labs(title = paste("Degree Aggregation (", region_name, ")"))
  
  cat("[INFO] Plotting variable importance...
")
  print(p_imp)
  cat("[INFO] Plotting degree aggregation...
")
  print(p_deg)
  
  return(list(vi = vi, plot_importance = p_imp, plot_degree = p_deg))
}

# ========================
# Test on One Region
# ========================

# Train and evaluate model for the ohmic region
ohmic_result <- train_model_region("ohmic")

# Show metrics
print("Performance metrics for 'ohmic' region:")
print(ohmic_result$metrics)

# Run glex and visualize explanations
ohmic_expl <- explain_model_glex("ohmic", ohmic_result$model)

# View feature importance values (data frame)
print("Top variable importance terms for 'ohmic':")
print(head(ohmic_expl$vi[order(-abs(ohmic_expl$vi$m)), ], 10))
