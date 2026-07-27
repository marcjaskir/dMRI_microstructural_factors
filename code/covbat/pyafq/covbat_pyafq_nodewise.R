
repo_root <- normalizePath("../..", mustWork = FALSE)
config_path <- Sys.getenv("DMRI_MICRO_CONFIG", file.path(repo_root, "config.yaml"))
if (!file.exists(config_path)) config_path <- file.path(repo_root, "config.example.yaml")
if (!requireNamespace("yaml", quietly = TRUE)) stop("Install R package yaml")
cfg <- yaml::read_yaml(config_path)
base_dir <- cfg$project_root
run_covbat_harmonization <- function(covbat_inputs_dir, tract, train_prop = 0.9) {

  library(ComBatFamily)
  library(mgcv)
  library(jsonlite)

  # Define outputs directory as same as inputs directory, but with the inputs directory swapped with outputs
  covbat_outputs_dir <- gsub("inputs", "outputs", covbat_inputs_dir)
  covbat_outputs_dir <- paste0(covbat_outputs_dir, "/", tract)

  # Make output directory if it doesn't exist
  if (!dir.exists(covbat_outputs_dir)) {
    dir.create(covbat_outputs_dir, recursive = TRUE)
  }

  # Load valid measures from JSON
  valid_measures <- names(fromJSON("base_dir/data/metadata/scalar_labels_to_filenames.json"))

  for (measure in valid_measures) {

    # Skip the measure map_li
    # Skip excluded scalars as listed in project context
    EXCLUDED_SCALARS <- c(
      "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz",
      "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2"
    )
    if (measure %in% EXCLUDED_SCALARS) {
      next
    }

    # Bat and covar are shared for mean and standard_deviation
    bat_fpath <- file.path(covbat_inputs_dir, tract, paste0(tract, "_", measure, "_bat.csv"))
    covar_fpath <- file.path(covbat_inputs_dir, tract, paste0(tract, "_", measure, "_covar.csv"))
    data_mean_path <- file.path(covbat_inputs_dir, tract, paste0(tract, "_", measure, "_mean_data.csv"))
    if (!file.exists(bat_fpath) || !file.exists(covar_fpath) || !file.exists(data_mean_path)) {
      next
    }
    bat_orig <- read.csv(bat_fpath)
    covar_orig <- read.csv(covar_fpath)
    data_for_split <- read.csv(data_mean_path)

    # Stratified train/test split once per measure (same split for mean and standard_deviation)
    set.seed(052525)
    covar_orig$split <- NA
    unique_bats <- unique(bat_orig$bat)
    for (b in unique_bats) {
      idx <- which(bat_orig$bat == b)
      n <- length(idx)
      n_train <- floor(train_prop * n)
      train_idx <- sample(idx, n_train)
      test_idx <- setdiff(idx, train_idx)
      covar_orig$split[train_idx] <- "train"
      covar_orig$split[test_idx] <- "test"
    }
    missing_idx <- which(rowSums(is.na(data_for_split)) > 0)
    if (length(missing_idx) > 0) {
      covar_orig$split[missing_idx] <- "test"
    }

    # Run CovBat for mean and for standard_deviation separately (same bat, covar, split)
    for (stat in c("mean", "standard_deviation")) {
      data_fname <- if (stat == "mean") {
        paste0(tract, "_", measure, "_mean_data.csv")
      } else {
        paste0(tract, "_", measure, "_standard_deviation_data.csv")
      }
      out_fname <- if (stat == "mean") {
        paste0(tract, "_", measure, "_stat-mean_covbat.csv")
      } else {
        paste0(tract, "_", measure, "_stat-standard_deviation_covbat.csv")
      }
      data_path <- file.path(covbat_inputs_dir, tract, data_fname)
      if (!file.exists(data_path)) {
        next
      }

      print(paste0("-- ", measure, " (", stat, ")"))

      data <- read.csv(data_path)

    # Process batch and covariate data
    scanner_id_levels <- unique(bat_orig$bat)
    bat <- factor(bat_orig$bat, levels = scanner_id_levels)

    # Convert covar_orig to data frame
    covar <- data.frame(covar_orig)
    
    # Get train/test indices
    train_indices <- which(covar$split == "train")
    test_indices <- which(covar$split == "test")

    # Get subject IDs for train/test sets
    sub_train <- covar$sub[train_indices]
    sub_test <- covar$sub[test_indices]

    # Split data, batch, and covariates into train/test sets
    data_train <- data[train_indices, -which(names(data) == "sub"), drop = FALSE]
    data_test <- data[test_indices, -which(names(data) == "sub"), drop = FALSE]

    bat_train <- bat[train_indices]
    bat_test <- bat[test_indices]
    
    covar_train <- covar[train_indices, -which(names(covar) %in% c("sub", "split")), drop = FALSE]
    covar_test <- covar[test_indices, -which(names(covar) %in% c("sub", "split")), drop = FALSE]

    # Initialize matrices to store harmonized train and test data
    harmonized_train_matrix <- matrix(NA, nrow = nrow(data_train), ncol = 100)
    harmonized_test_matrix <- matrix(NA, nrow = nrow(data_test), ncol = 100)

    # Iterate over 1-100 (nodes)
    for (node in 1:100) {

      # Subset train/test data for current measure
      data_measure_train <- data_train[, paste0(measure, "_node", node), drop = FALSE]
      data_measure_test <- data_test[, paste0(measure, "_node", node), drop = FALSE]

      # Run CovBat harmonization on training data
      formula <- y ~ s(age, k = 3, fx = TRUE) + sex

      model <- covfam(
        data = data_measure_train,
        bat = bat_train,
        covar = covar_train,
        mod = gam,
        formula = formula,
      )
      harmonized_train_matrix[, node] <- model$dat.covbat

      # Apply harmonization to test data
      test_apply <- predict(
        model,
        newdata = data_measure_test,
        newbat = bat_test,
        newcovar = covar_test
      )
      harmonized_test_matrix[, node] <- test_apply$dat.covbat

    }

    # Convert matrices to data frames, where columns are {measure}_node{node}
    harmonized_train_df <- data.frame(harmonized_train_matrix)
    colnames(harmonized_train_df) <- paste0(measure, "_node", 1:100)
    harmonized_test_df <- data.frame(harmonized_test_matrix)
    colnames(harmonized_test_df) <- paste0(measure, "_node", 1:100)
    
    # Merge in metadata for train
    harmonized_train_metadata <- cbind(sub_train, bat_train, covar_train)
    harmonized_train_metadata$split <- "train"
    harmonized_train_df <- cbind(harmonized_train_metadata, harmonized_train_df)
    colnames(harmonized_train_df)[colnames(harmonized_train_df) == "sub_train"] <- "sub"
    colnames(harmonized_train_df)[colnames(harmonized_train_df) == "bat_train"] <- "bat"
    
    # Merge in metadata for test
    harmonized_test_metadata <- cbind(sub_test, bat_test, covar_test)
    harmonized_test_metadata$split <- "test"
    harmonized_test_df <- cbind(harmonized_test_metadata, harmonized_test_df)
    colnames(harmonized_test_df)[colnames(harmonized_test_df) == "sub_test"] <- "sub"
    colnames(harmonized_test_df)[colnames(harmonized_test_df) == "bat_test"] <- "bat"
    
    # Combine harmonized train and test data
    harmonized_data <- rbind(harmonized_train_df, harmonized_test_df)

    # Sort by group and then by sub
    harmonized_data <- harmonized_data[order(harmonized_data$group, harmonized_data$sub), ]
    
    # Save harmonized data to CSV (stat-specific: _stat-mean_covbat.csv or _stat-standard_deviation_covbat.csv)
    write.csv(
      harmonized_data,
      file.path(covbat_outputs_dir, out_fname),
      row.names = FALSE,
      quote = FALSE
    )
    }  # end for (stat in c("mean", "standard_deviation"))
  }  # end for (measure in valid_measures)
}

wm_atlas <- "HCP1065"
print(wm_atlas)

covbat_inputs_dir <- paste0("base_dir/derivatives/covbat/inputs/pyafq/", wm_atlas)
covbat_outputs_dir <- paste0("base_dir/derivatives/covbat/outputs/pyafq/", wm_atlas)
tracts <- sapply(list.dirs(covbat_inputs_dir, full.names = FALSE, recursive = FALSE), basename)

for (tract in tracts) {

  # If the output directory already exists, skip
  if (dir.exists(paste0(covbat_outputs_dir, "/", tract))) {
    next
  }

  print(paste0("Running CovBat for ", tract))
  run_covbat_harmonization(covbat_inputs_dir, tract)
}