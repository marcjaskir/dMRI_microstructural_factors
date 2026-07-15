
repo_root <- normalizePath("../..", mustWork = FALSE)
config_path <- Sys.getenv("DMRI_MICRO_CONFIG", file.path(repo_root, "config.yaml"))
if (!file.exists(config_path)) config_path <- file.path(repo_root, "config.example.yaml")
if (!requireNamespace("yaml", quietly = TRUE)) stop("Install R package yaml")
cfg <- yaml::read_yaml(config_path)
base_dir <- cfg$project_root
require(mgcv)
require(jsonlite)

wm_atlas <- "HCP1065"

covbat_outputs_dir <- file.path(base_dir, "derivatives/covbat/outputs/pyafq", wm_atlas)
covbat_inputs_dir <- file.path(base_dir, "derivatives/covbat/inputs/pyafq", wm_atlas)
gam_outputs_dir <- file.path(base_dir, "derivatives/gam/pyafq", wm_atlas)

CONTROL_GROUPS <- c("penn_controls", "hcpya", "hcpaging")

run_gam <- function(wm_atlas,
                    tract,
                    covbat_outputs_dir,
                    covbat_inputs_dir,
                    gam_outputs_dir,
                    measures_json_path = file.path(base_dir, "data/metadata/scalar_labels_to_filenames.json")) {

  valid_measures <- names(fromJSON(measures_json_path))

  for (measure in valid_measures) {

    EXCLUDED_MEASURES <- c(
      "map_li", "map_am", "dti_txx", "dti_txy", "dti_txz",
      "dti_tyy", "dti_tyz", "dti_tzz", "dti_ha", "rdi_rd1", "rdi_rd2", "gqi_iso"
    )
    if (measure %in% EXCLUDED_MEASURES) {
      next
    }

    for (stat in c("mean", "standard_deviation")) {
      stat_suffix <- if (stat == "mean") "stat-mean" else "stat-standard_deviation"
      covbat_fname <- paste0(tract, "_", measure, "_", stat_suffix, "_covbat.csv")
      gam_fname <- paste0(tract, "_", measure, "_", stat_suffix, "_gam.csv")

      if (file.exists(file.path(gam_outputs_dir, tract, gam_fname))) {
        print(paste0("Z-score file already exists for ", tract, " ", measure, " (", stat, ")"))
        next
      }

      covbat_file <- file.path(covbat_outputs_dir, tract, covbat_fname)
      if (!file.exists(covbat_file)) {
        next
      }

      data_fname <- if (stat == "mean") {
        paste0(tract, "_", measure, "_mean_data.csv")
      } else {
        paste0(tract, "_", measure, "_standard_deviation_data.csv")
      }
      unharm_data_path <- file.path(covbat_inputs_dir, tract, data_fname)
      if (!file.exists(unharm_data_path)) {
        warning("Pre-CovBat scalar file not found, skipping: ", unharm_data_path)
        next
      }

      print(paste0("-- ", measure, " (", stat, ")"))

      covbat_df <- read.csv(covbat_file, stringsAsFactors = FALSE)
      d1_unharm <- read.csv(unharm_data_path, stringsAsFactors = FALSE)

      demographic_cols <- grep(paste0(measure, "_node"), colnames(covbat_df), value = TRUE, invert = TRUE)
      covbat_demo_df <- covbat_df[, demographic_cols, drop = FALSE]

      if (!dir.exists(file.path(gam_outputs_dir, tract))) {
        dir.create(file.path(gam_outputs_dir, tract), recursive = TRUE)
      }

      gam_data <- data.frame(sub = covbat_df$sub, stringsAsFactors = FALSE)

      controls_mask <- covbat_df$group %in% CONTROL_GROUPS

      for (node in 1:100) {
        node_col <- paste0(measure, "_node", node)
        unharm_name <- paste0("node", node, "_unharm")

        if (!node_col %in% names(d1_unharm)) {
          warning("Column ", node_col, " missing in ", unharm_data_path, "; using NA for unharmonized node ", node)
          unharm_vec <- rep(NA_real_, nrow(covbat_df))
        } else {
          unharm_vec <- d1_unharm[[node_col]][match(as.character(covbat_df$sub), as.character(d1_unharm$sub))]
        }

        formula_h <- as.formula(paste0(node_col, " ~ s(age, k=3, fx=TRUE) + sex"))
        controls_hs <- covbat_df[controls_mask, , drop = FALSE]
        gam_model <- gam(formula_h, data = controls_hs)

        controls_residuals <- residuals(gam_model)
        sd_control <- sd(controls_residuals)

        all_pred <- predict(gam_model, newdata = covbat_df)
        all_residuals <- covbat_df[[node_col]] - all_pred

        gam_data[[paste0("node", node)]] <- covbat_df[[node_col]]
        gam_data[[paste0("node", node, "_pred")]] <- all_pred
        gam_data[[paste0("node", node, "_z")]] <- all_residuals / sd_control
        gam_data[[paste0("node", node, "_centile")]] <- ecdf(controls_residuals)(all_residuals) * 100

        gam_data[[unharm_name]] <- unharm_vec

        covbat_df[[unharm_name]] <- unharm_vec
        controls_u <- covbat_df[controls_mask, , drop = FALSE]
        controls_u <- controls_u[
          !is.na(controls_u[[unharm_name]]) & !is.na(controls_u$age) & !is.na(controls_u$sex),
          ,
          drop = FALSE
        ]
        if (nrow(controls_u) < 10L) {
          warning("Too few controls with non-NA unharmonized data for ", tract, " ", measure, " node ", node)
          unharm_pred <- rep(NA_real_, nrow(covbat_df))
        } else {
          formula_u <- as.formula(paste0(unharm_name, " ~ s(age, k=3, fx=TRUE) + sex"))
          gam_unharm <- gam(formula_u, data = controls_u)
          unharm_pred <- as.numeric(predict(gam_unharm, newdata = covbat_df))
        }
        covbat_df[[unharm_name]] <- NULL

        gam_data[[paste0("node", node, "_unharm_pred")]] <- unharm_pred
      }

      gam_data <- merge(covbat_demo_df, gam_data, by = "sub")

      write.csv(
        gam_data,
        file.path(gam_outputs_dir, tract, gam_fname),
        row.names = FALSE,
        quote = FALSE
      )
    }
  }
}

tracts <- list.files(covbat_outputs_dir)

for (tract in tracts) {
  print(paste0("Running GAM for ", tract))
  run_gam(wm_atlas, tract, covbat_outputs_dir, covbat_inputs_dir, gam_outputs_dir)
}
