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
#       known rules (V < 0 or > 1.23 V for first 5 points, non-monotonic behavior)
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
data_pc = read.csv("../../data/raw/data_for_classification_up_until_270525.csv", stringsAsFactors = TRUE)
data_pc$id = NULL

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
  Sa          = 1.2,       # Anode stoichiometry
  Sc          = 2.0,       # Cathode stoichiometry
  Phi_a_des   = 0.4,       # Desired RH anode
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

#--- define task and model ----
task = TaskClassif$new(id = "pem", backend = data_pc, target = "validity")

# use a random forest, we want probabilities out of it
mod = lrn("classif.ranger", predict_type = "prob")
set.seed(42)
mod$train(task)

#--- create predictor object ----
pred = Predictor$new(model = mod,
                     data = data_pc,
                     y = "validity",
                     type = "classification",
                     class = "valid")

#--- apply PRIM to find a valid zone ----
prim = Prim$new(predictor = pred)
prim_box = prim$find_box(x_interest = x_interest, desired_range = c(0.9, 1.0))  # we want mostly valid

# evaluate and see how clean that region is
prim_box$evaluate()

#--- postprocess the box to make it tighter
postproc = PostProcessing$new(predictor = pred)
post_box = postproc$find_box(x_interest = x_interest,
                             desired_range = c(0.9, 1.0),
                             box_init = prim_box$box)

print(postproc)
# check how it did
post_box$evaluate()

# optional: plot a 2D slice
post_box$plot_surface(feature_names = c("T", "RH"), surface = "range")  # just two vars you care about

