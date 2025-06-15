#---------------------------------------------------------------------------
#  Interpretable Regional Descriptors for Valid PEMFC Polarization Curves
# --------------------------------------------------------------------------
# Goal:
# Use interpretable regional descriptor methods (e.g., PRIM, MaxBox, MAIRE, Anchors)
# to identify a hyperbox in the input space of the AlphaPEM model where
# the polarization curves are consistently valid.
#
# Input:
# - A data frame containing:
#     - 19 input variables (undetermined physical params + operating conditions)
#     - A binary classification label: "valid" or "invalid" based on 
#       known rules (V < 0 or > 1.23 V for first points, strong non-monotonic behavior)
#
# Approach:
# - Train a classifier (e.g., random forest) to predict validity
# - Use a regional descriptor method to find an interpretable subspace
#   (hyperbox) that captures the region of valid behavior
# - Sample only within this hyperbox for future experiments
#
# Expected Output:
# - Interpretable constraints (intervals) on each variable
# - A regional descriptor object that defines a valid sampling region
# --------------------------------------------------------------------------

#--- setup ----
setwd("../external/irdpackage")

library("mlr3")
library("mlr3learners")
library("iml")
library("mlr3pipelines")
library("devtools")

load_all()

#--- load the data ----
data_pc = read.csv("../../data/processed/configurations_for_IRD_until_2025-06-15.csv",
                   stringsAsFactors = TRUE)

# data_pc$id = NULL

View(data_pc)

# rename the target col to a more specific name
names(data_pc)[names(data_pc) == "classification"] = "validity"

# just making sure it's treated as a factor
data_pc$validity = factor(data_pc$validity, levels = c("invalid", "valid"))

#--- define x_interest manually using known calibrated values ----

# Values extracted from tables in the thesis -> table 3.3 and 3.4

x_interest = data.frame(
  Tfc         = 347.15,    # Cell temperature [K]
  Pa_des      = 2.0*1e5,   # Anode pressure [Pascal]
  Pc_des      = 2.0*1e5,   # Cathode pressure [Pascal]
  Sa          = 1.3,       # Anode stoichiometry (CHANGED from 1.2)
  Sc          = 2.0,       # Cathode stoichiometry
  Phi_a_des   = 0.5,       # Desired RH anode (CHANGED from 0.4)
  Phi_c_des   = 0.6,       # Desired RH cathode
  epsilon_gdl = 0.701,     # GDL porosity
  tau         = 1.02,      # Pore structure coefficient
  epsilon_mc  = 0.399,     # Ionomer volume fraction in CLs
  epsilon_c   = 0.271,     # GDL compression ratio
  e           = 5.0,       # Capillary exponent
  Re          = 5.7e-7,    # Electron conduction resistance [Ω·m²]
  i0_c_ref    = 2.79,      # Ref. cathode exchange current density [A/m²]
  kappa_co    = 27.2,      # Crossover correction coefficient
  kappa_c     = 1.61,      # Overpotential correction exponent
  a_slim      = 0.056,     # s_lim coefficient (bar⁻¹)
  b_slim      = 0.105,     # s_lim coefficient (dimensionless)
  a_switch    = 0.637      # s_lim switching point
)

# add dummy label to match task structure (won't affect box finding)
x_interest$validity = factor("valid", levels = c("invalid", "valid"))

#----------------------------------------------------------------
#  Clean the df to contain only cols in x_interest + target
#----------------------------------------------------------------
my_target = "validity"

data_pc <- data_pc[, names(x_interest), drop = FALSE]

#----------------------------------------------------------------
#     Fit a RF to classify between valid and invalid
#----------------------------------------------------------------

#--- define classification task ----
task = TaskClassif$new(id = "pem", backend = data_pc, target = my_target)

#--- train random forest classifier ----
mod = lrn("classif.ranger", predict_type = "prob")
set.seed(42)
mod$train(task)

#--- evaluate classifier (train/test split) ----
train_idx = sample(task$nrow, 0.8 * task$nrow)
test_idx = setdiff(seq_len(task$nrow), train_idx)
task_train = task$clone()$filter(train_idx)
task_test  = task$clone()$filter(test_idx)

mod$train(task_train)
prediction = mod$predict(task_test)

prediction$score(msrs(c("classif.acc", "classif.precision", "classif.recall", "classif.fbeta", "classif.auc")))

#--- wrap model in Predictor for IRD methods ----
pred = Predictor$new(model = mod,
                     data = data_pc,
                     y = my_target,
                     type = "classification",
                     class = "valid")

#----------------------------------------------------------------
#       Try out different IRD methods
#----------------------------------------------------------------
my_prob_range = c(0.8, 1.0)
#----------------------------------------------------------------
# Option 1: apply PRIM to find a valid zone 
#----------------------------------------------------------------

prim = Prim$new(predictor = pred)
prim_box = prim$find_box(x_interest = x_interest, desired_range = my_prob_range)  # we want mostly valid

prim_box$evaluate()  # initial evaluation

# Okay: Box has an impurity of 0.01
# but it's relatively close from x_interest (dist approx. 0.052) 

#--- postprocess PRIM box ----
postproc_prim = PostProcessing$new(predictor = pred)
post_box_prim = postproc_prim$find_box(x_interest = x_interest,
                                       desired_range = my_prob_range,
                                       box_init = prim_box$box)

post_box_prim$evaluate()
post_box_prim

#post_box_prim$plot_surface(feature_names = c("Tfc", "Pa_des"), surface = "range")


#----------------------------------------------------------------
# Option 2: compute regional descriptor with MaxBox 
#----------------------------------------------------------------
set.seed(42)

mb = MaxBox$new(predictor = pred, quiet = FALSE, strategy = "traindata")
system.time({
  mbb = mb$find_box(x_interest = x_interest, desired_range = my_prob_range)
})

mbb
mbb$evaluate()

# mbb$plot_surface(feature_names = c("Tfc", "Pa_des"), surface = "range")

#--- postprocess MaxBox box to refine ----
postproc_maxbox = PostProcessing$new(predictor = pred)
post_box_maxbox = postproc_maxbox$find_box(x_interest = x_interest,
                                           desired_range = my_prob_range,
                                           box_init = mbb$box)

post_box_maxbox$evaluate()
post_box_maxbox

#post_box_maxbox$plot_surface(feature_names = c("Tfc", "Pa_des"), surface = "range")


#----------------------------------------------------------------
# Option 3: compute regional descriptor with MAIRE box
#----------------------------------------------------------------

tensorflow::tf$compat$v1$disable_eager_execution()

mair = Maire$new(predictor = pred,
                 num_of_iterations = 100L,
                 convergence = TRUE,
                 quiet = FALSE,
                 strategy = "traindata")

system.time({
  mairb = mair$find_box(x_interest = x_interest, desired_range = my_prob_range)
})

mairb$evaluate()
mairb$plot_surface(feature_names = c("Tfc", "Pa_des"), surface = "range")

#--- postprocess MAIRE box ----
postproc_maire = PostProcessing$new(predictor = pred, subbox_relsize = 0.3)
post_box_maire = postproc_maire$find_box(x_interest = x_interest,
                                         desired_range = my_prob_range,
                                         box_init = mairb$box)

post_box_maire$evaluate()
print(post_box_maire)

# post_box_maire$plot_surface(feature_names = c("Tfc", "Pa_des"), surface = "range")

#----------------------------------------------------------------
#   Save the results for a given method
#----------------------------------------------------------------

extract_ird_bounds <- function(post_box_obj) {
  # Extract lower and upper bounds
  lower_bounds <- post_box_obj$box$lower
  upper_bounds <- post_box_obj$box$upper
  
  # Create a data frame
  bounds_df <- data.frame(
    parameter = names(lower_bounds),
    lower = as.numeric(lower_bounds),
    upper = as.numeric(upper_bounds)
  )
  
  return(bounds_df)
}

ranges_prim <- extract_ird_bounds(post_box_prim)
View(ranges_prim)

write.csv(ranges_prim,
          "../../data/processed/hyperbox_bounds/hyperbox_prim_150625.csv",
          row.names = FALSE)
