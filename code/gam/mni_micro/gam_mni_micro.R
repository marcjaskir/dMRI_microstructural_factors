
repo_root <- normalizePath("../..", mustWork = FALSE)
config_path <- Sys.getenv("DMRI_MICRO_CONFIG", file.path(repo_root, "config.yaml"))
if (!file.exists(config_path)) config_path <- file.path(repo_root, "config.example.yaml")
if (!requireNamespace("yaml", quietly = TRUE)) stop("Install R package yaml")
cfg <- yaml::read_yaml(config_path)
base_dir <- cfg$project_root
require(mgcv)
require(jsonlite)

covbat_outputs_base <- file.path(base_dir, "derivatives/covbat/outputs/mni_micro")
covbat_inputs_base <- file.path(base_dir, "derivatives/covbat/inputs/mni_micro")
gam_outputs_base <- file.path(if (!is.null(cfg$gam_dir)) cfg$gam_dir else file.path(base_dir, "derivatives/gam/mni_micro")

# Atlases to process: 4S156 (GM parcels), HCP1065 (mni_micro tract segments, if present)
atlases <- c("Glasser")

#' Read CovBat **input** scalar (before harmonization) merged on ``sub`` with covariates,
#' aligned to ``covbat_df$sub`` order. Returns a numeric vector of length ``nrow(covbat_df)``.
load_unharmonized_scalar <- function(roi, measure, stat, covbat_df, covbat_inputs_dir) {
  covar_path <- file.path(covbat_inputs_dir, roi, paste0(roi, "_", measure, "_covar.csv"))
  if (!file.exists(covar_path)) {
    stop("Covar input not found: ", covar_path)
  }
  data_fname <- if (stat == "mean") {
    paste0(roi, "_", measure, "_mean_data.csv")
  } else if (stat == "standard_deviation") {
    paste0(roi, "_", measure, "_standard_deviation_data.csv")
  } else {
    stop("Unknown stat: ", stat)
  }
  data_path <- file.path(covbat_inputs_dir, roi, data_fname)
  if (!file.exists(data_path)) {
    stop("Scalar input not found: ", data_path)
  }
  d1 <- read.csv(data_path, stringsAsFactors = FALSE)
  d2 <- read.csv(covar_path, stringsAsFactors = FALSE)
  if (!measure %in% names(d1)) {
    stop("Column ", measure, " not in ", data_path)
  }
  merged_in <- merge(d1[, c("sub", measure)], d2, by = "sub")
  v <- merged_in[[measure]][match(as.character(covbat_df$sub), as.character(merged_in$sub))]
  v
}

run_gam <- function(atlas,
                    roi,
                    covbat_outputs_dir,
                    covbat_inputs_dir,
                    gam_outputs_dir,
                    measures_json_path = "base_dir/data/metadata/scalar_labels_to_filenames.json") {

  valid_measures <- names(fromJSON(measures_json_path))

  for (measure in valid_measures) {

    if (measure %in% c("map_li", "gqi_iso")) {
      next
    }

    for (stat in c("mean", "standard_deviation")) {

      covbat_file <- file.path(
        covbat_outputs_dir,
        roi,
        paste0(roi, "_", measure, "_stat-", stat, "_covbat.csv")
      )

      output_name <- paste0(roi, "_", measure, "_stat-", stat, "_gam.csv")
      zscore_check_file <- file.path(gam_outputs_dir, roi, paste0(roi, "_", measure, "_stat-", stat, "_gam.csv"))

      if (file.exists(zscore_check_file)) {
        print(paste0("Z-score file already exists for ", roi, " ", measure, " stat-", stat))
        next
      }

      if (!file.exists(covbat_file)) {
        print(paste0("CovBat file not found for ", roi, " ", measure, " stat-", stat))
        next
      }

      print(paste0("-- ", measure, " [stat-", stat, "]"))

      covbat_df <- read.csv(covbat_file, stringsAsFactors = FALSE)

      demographic_cols <- c("sub", "bat", "age", "sex", "group", "split")
      covbat_demo_df <- covbat_df[, demographic_cols, drop = FALSE]

      unharm_name <- paste0(measure, "_unharm")
      unharm_pred_name <- paste0(measure, "_unharm_pred")

      # --- Pre-CovBat (unharmonized) scalar, aligned to harmonized CovBat rows ---
      unharm_vec <- tryCatch(
        load_unharmonized_scalar(roi, measure, stat, covbat_df, covbat_inputs_dir),
        error = function(e) {
          warning(conditionMessage(e))
          NULL
        }
      )
      if (is.null(unharm_vec)) {
        print(paste0("Skipping GAM for ", roi, " ", measure, " stat-", stat, " (could not load unharmonized inputs)"))
        next
      }

      covbat_df[[unharm_name]] <- unharm_vec

      if (!dir.exists(file.path(gam_outputs_dir, roi))) {
        dir.create(file.path(gam_outputs_dir, roi), recursive = TRUE)
      }

      gam_data <- data.frame(sub = covbat_df$sub, stringsAsFactors = FALSE)

      controls_data <- covbat_df[covbat_df$group %in% c("penn_controls", "hcpya", "hcpaging"), , drop = FALSE]

      # ---- GAM on harmonized (CovBat) scalar ----
      formula_h <- as.formula(paste0(measure, " ~ s(age, k=3, fx=TRUE) + sex"))
      gam_model <- gam(formula_h, data = controls_data)

      controls_residuals <- residuals(gam_model)
      sd_control <- sd(controls_residuals)

      all_pred <- predict(gam_model, newdata = covbat_df)
      all_residuals <- covbat_df[[measure]] - all_pred

      gam_data[[measure]] <- covbat_df[[measure]]
      gam_data[[paste0(measure, "_pred")]] <- all_pred
      gam_data[[paste0(measure, "_z")]] <- all_residuals / sd_control
      gam_data[[paste0(measure, "_centile")]] <- ecdf(controls_residuals)(all_residuals) * 100

      # Unharmonized values and GAM fit to unharmonized controls
      gam_data[[unharm_name]] <- unharm_vec

      controls_u <- controls_data[!is.na(controls_data[[unharm_name]]), , drop = FALSE]
      if (nrow(controls_u) < 10L) {
        warning("Too few controls with non-NA unharmonized data for ", roi, " ", measure)
        gam_data[[unharm_pred_name]] <- rep(NA_real_, nrow(covbat_df))
      } else {
        formula_u <- as.formula(paste0(unharm_name, " ~ s(age, k=3, fx=TRUE) + sex"))
        gam_unharm <- gam(formula_u, data = controls_u)
        gam_data[[unharm_pred_name]] <- as.numeric(predict(gam_unharm, newdata = covbat_df))
      }

      gam_data <- merge(covbat_demo_df, gam_data, by = "sub")

      write.csv(
        gam_data,
        file.path(gam_outputs_dir, roi, output_name),
        row.names = FALSE,
        quote = FALSE
      )
    }
  }
}

for (atlas in atlases) {
  covbat_outputs_dir <- file.path(covbat_outputs_base, atlas)
  covbat_inputs_dir <- file.path(covbat_inputs_base, atlas)
  gam_outputs_dir <- file.path(gam_outputs_base, atlas)

  if (!dir.exists(covbat_outputs_dir)) {
    message("CovBat outputs not found for atlas ", atlas, ", skipping")
    next
  }

  rois <- list.files(covbat_outputs_dir)
  if (length(rois) == 0) {
    message("No ROIs in ", covbat_outputs_dir, ", skipping")
    next
  }

  message("Atlas: ", atlas, " (", length(rois), " ROIs)")
  for (roi in rois) {
    roi_path <- file.path(covbat_outputs_dir, roi)
    if (!dir.exists(roi_path)) next
    print(paste0("Running GAM for ", atlas, " / ", roi))
    run_gam(atlas, roi, covbat_outputs_dir, covbat_inputs_dir, gam_outputs_dir)
  }
}
