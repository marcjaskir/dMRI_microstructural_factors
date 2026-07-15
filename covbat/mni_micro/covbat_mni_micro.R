
repo_root <- normalizePath("../..", mustWork = FALSE)
config_path <- Sys.getenv("DMRI_MICRO_CONFIG", file.path(repo_root, "config.yaml"))
if (!file.exists(config_path)) config_path <- file.path(repo_root, "config.example.yaml")
if (!requireNamespace("yaml", quietly = TRUE)) stop("Install R package yaml")
cfg <- yaml::read_yaml(config_path)
base_dir <- cfg$project_root
# CovBat harmonization for mni_micro: reads inputs from
# derivatives/covbat/inputs/mni_micro/{atlas}/{parcel}/ and writes
# derivatives/covbat/outputs/mni_micro/{atlas}/{glasser_label}/{glasser_label}_{scalar}_{stat}_covbat.csv
# for stat in (mean, standard_deviation).

run_covbat_harmonization <- function(covbat_inputs_dir, parcel, scalar, stat, atlas, train_prop = 0.9, glasser_label = NULL) {
  library(ComBatFamily)
  library(mgcv)

  # By default, the label for output is the same as the parcel
  label_for_output <- parcel
  if (!is.null(glasser_label) && atlas == "Glasser") {
    label_for_output <- glasser_label
  }

  parcel_dir <- file.path(covbat_inputs_dir, parcel)
  covbat_outputs_dir <- gsub("inputs", "outputs", covbat_inputs_dir)
  # Only change the output directory structure for Glasser if glasser_label is provided
  if (!is.null(glasser_label) && atlas == "Glasser") {
    covbat_outputs_dir <- file.path(covbat_outputs_dir, label_for_output)
  } else {
    covbat_outputs_dir <- file.path(covbat_outputs_dir, parcel)
  }

  out_path <- file.path(covbat_outputs_dir, paste0(label_for_output, "_", scalar, "_stat-", stat, "_covbat.csv"))
  if (file.exists(out_path)) return(invisible(NULL))

  if (!dir.exists(covbat_outputs_dir)) {
    dir.create(covbat_outputs_dir, recursive = TRUE)
  }

  data_path <- file.path(parcel_dir, paste0(parcel, "_", scalar, "_", stat, "_data.csv"))
  bat_path <- file.path(parcel_dir, paste0(parcel, "_", scalar, "_bat.csv"))
  covar_path <- file.path(parcel_dir, paste0(parcel, "_", scalar, "_covar.csv"))

  if (!file.exists(data_path) || !file.exists(bat_path) || !file.exists(covar_path)) {
    return(invisible(NULL))
  }

  data <- read.csv(data_path)
  bat_orig <- read.csv(bat_path)
  covar_orig <- read.csv(covar_path)

  # Outcome column is the scalar name
  measure <- scalar

  set.seed(052525)
  covar_orig$split <- NA
  unique_bats <- unique(bat_orig$bat)
  bat <- factor(bat_orig$bat, levels = unique_bats)

  for (b in unique_bats) {
    idx <- which(bat_orig$bat == b)
    n <- length(idx)
    n_train <- floor(train_prop * n)
    train_idx <- sample(idx, n_train)
    test_idx <- setdiff(idx, train_idx)
    covar_orig$split[train_idx] <- "train"
    covar_orig$split[test_idx] <- "test"
  }

  missing_idx <- which(rowSums(is.na(data)) > 0)
  if (length(missing_idx) > 0) {
    covar_orig$split[missing_idx] <- "test"
  }

  scanner_id_levels <- unique(bat_orig$bat)
  bat <- factor(bat_orig$bat, levels = scanner_id_levels)
  covar <- data.frame(covar_orig)

  train_indices <- which(covar$split == "train")
  test_indices <- which(covar$split == "test")

  sub_train <- covar$sub[train_indices]
  sub_test <- covar$sub[test_indices]

  data_train <- data[train_indices, -which(names(data) == "sub"), drop = FALSE]
  data_test <- data[test_indices, -which(names(data) == "sub"), drop = FALSE]
  bat_train <- bat[train_indices]
  bat_test <- bat[test_indices]
  covar_train <- covar[train_indices, -which(names(covar) %in% c("sub", "split")), drop = FALSE]
  covar_test <- covar[test_indices, -which(names(covar) %in% c("sub", "split")), drop = FALSE]

  data_measure_train <- data_train[, measure, drop = FALSE]
  data_measure_test <- data_test[, measure, drop = FALSE]

  formula <- y ~ s(age, k = 3, fx = TRUE) + sex
  model <- covfam(
    data = data_measure_train,
    bat = bat_train,
    covar = covar_train,
    mod = gam,
    formula = formula,
  )

  test_apply <- predict(
    model,
    newdata = data_measure_test,
    newbat = bat_test,
    newcovar = covar_test
  )

  harmonized_train_df <- data.frame(model$dat.covbat)
  harmonized_test_df <- data.frame(test_apply$dat.covbat)
  colnames(harmonized_train_df) <- measure
  colnames(harmonized_test_df) <- measure

  harmonized_train_metadata <- cbind(sub_train, bat_train, covar_train)
  harmonized_train_metadata$split <- "train"
  harmonized_train_df <- cbind(harmonized_train_metadata, harmonized_train_df)
  colnames(harmonized_train_df)[colnames(harmonized_train_df) == "sub_train"] <- "sub"
  colnames(harmonized_train_df)[colnames(harmonized_train_df) == "bat_train"] <- "bat"

  harmonized_test_metadata <- cbind(sub_test, bat_test, covar_test)
  harmonized_test_metadata$split <- "test"
  harmonized_test_df <- cbind(harmonized_test_metadata, harmonized_test_df)
  colnames(harmonized_test_df)[colnames(harmonized_test_df) == "sub_test"] <- "sub"
  colnames(harmonized_test_df)[colnames(harmonized_test_df) == "bat_test"] <- "bat"

  harmonized_data <- rbind(harmonized_train_df, harmonized_test_df)
  harmonized_data <- harmonized_data[order(harmonized_data$group, harmonized_data$sub), ]

  write.csv(harmonized_data, out_path, row.names = FALSE, quote = FALSE)
  invisible(NULL)
}


# --- Main: discover atlases and parcels, run for each (parcel, scalar, stat) ---
covbat_inputs_dir <- "base_dir/derivatives/covbat/inputs/mni_micro"
valid_measures <- names(jsonlite::fromJSON("base_dir/data/metadata/scalar_labels_to_filenames.json"))
valid_measures <- setdiff(valid_measures, "map_li")

stats <- c("mean", "standard_deviation")

if (!dir.exists(covbat_inputs_dir)) {
  stop("CovBat inputs directory not found: ", covbat_inputs_dir)
}

atlases <- list.dirs(covbat_inputs_dir, full.names = FALSE, recursive = FALSE)

for (atlas in atlases) {

  # Skip unless atlas is HCP1065
  if (atlas != "HCP1065") {
    next
  }

  # Lookup table only for Glasser: index <-> label correspondence
  glasser_lookup <- NULL
  if (atlas == "Glasser") {
    glasser_lookup_path <- "base_dir/data/atlases/Glasser/atlas-Glasser_dseg.tsv"
    if (file.exists(glasser_lookup_path)) {
      glasser_lookup <- read.table(glasser_lookup_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
    } else {
      message("Skipping Glasser: cannot find lookup table at ", glasser_lookup_path)
      next
    }
  }
  # For 4S156 and HCP1065, no lookup: parcel name is used directly as output label

  atlas_input_dir <- file.path(covbat_inputs_dir, atlas)
  parcels <- list.dirs(atlas_input_dir, full.names = FALSE, recursive = FALSE)

  for (parcel in parcels) {
    parcel_dir <- file.path(atlas_input_dir, parcel)
    # Discover data files: {parcel}_{scalar}_{stat}_data.csv
    mean_files <- list.files(parcel_dir, pattern = "_mean_data\\.csv$", full.names = FALSE)
    std_files <- list.files(parcel_dir, pattern = "_standard_deviation_data\\.csv$", full.names = FALSE)

    # Glasser: resolve Parcel_{index} to label via lookup; 4S156/HCP1065: use parcel as label
    glasser_label <- NULL
    if (atlas == "Glasser" && !is.null(glasser_lookup) && grepl("^Parcel_[0-9]+$", parcel)) {
      idx <- as.numeric(gsub("Parcel_", "", parcel))
      if (!is.na(idx)) {
        matched <- glasser_lookup[glasser_lookup$index == idx, ]
        if (nrow(matched) > 0 && !is.na(matched$label[1])) {
          glasser_label <- matched$label[1]
        }
      }
    }

    for (f in c(mean_files, std_files)) {
      if (grepl("_mean_data\\.csv$", f)) {
        stat <- "mean"
        suffix <- "_mean_data.csv"
      } else {
        stat <- "standard_deviation"
        suffix <- "_standard_deviation_data.csv"
      }
      # f = "{parcel}_{scalar}_{stat}_data.csv"
      prefix <- paste0(parcel, "_")
      if (!startsWith(f, prefix) || !endsWith(f, suffix)) next
      scalar <- substring(f, nchar(prefix) + 1L, nchar(f) - nchar(suffix))
      if (!scalar %in% valid_measures) next

      print(paste0("CovBat: ", atlas, " / ", parcel, " / ", scalar, " / ", stat, if (!is.null(glasser_label)) paste0(" / ", glasser_label)))
      tryCatch(
        run_covbat_harmonization(atlas_input_dir, parcel, scalar, stat, atlas, glasser_label = glasser_label),
        error = function(e) message("Error: ", conditionMessage(e))
      )
    }
  }
}
