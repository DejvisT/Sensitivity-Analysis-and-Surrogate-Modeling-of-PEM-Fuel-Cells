############################################
### German Credit Application
############################################


# install.packages("anchors")
# install.packages("keras")
# install.packages("tensorflow")
#
# # Configure TensorFlow (it will install Python stuff too)
# library(keras)
# install_keras()                  # installs TensorFlow + Python backend

#--- setup ----
setwd("C:/Users/User/Documents/0. Semestres/SS2025/Consulting/supplementary_2023_ird-main/irdpackage")

library("mlr3")
library("mlr3learners")
library("iml")
library("mlr3pipelines")
library("devtools")

load_all()

###---- get data ----

credit = read.csv("inst/examples/data/german_credit_data.csv", row.names = 1, stringsAsFactors = TRUE)
names(credit)

# omit rows with NA entries
credit = na.omit(credit)

# join groups with small frequencies
levels(credit$Purpose) = c("others", "car", "others", "others",
  "furniture", "radio/TV", "others", "others")
levels(credit$Saving.accounts) = c("little", "moderate", "rich", "rich")
credit$Job = factor(credit$Job, levels = c(0, 1, 2, 3),
  labels = c("unskilled", "unskilled", "skilled", "highly skilled"))


# colnames to lower
names(credit) = tolower(names(credit))
# Drop levels
credit = droplevels.data.frame(credit)
credit$saving.accounts = as.ordered(credit$saving.accounts)
credit$checking.account = as.ordered(credit$checking.account)

###---- define x_interest and training data ----
x_interest = credit[1,]

# First use case: young woman buying a car
x_interest$credit.amount = 4000L
x_interest$purpose = factor("car", levels = levels(credit$purpose))
x_interest$housing = factor("rent", levels = levels(credit$housing))
x_interest$duration = 30L

#----- create task -----

task = TaskClassif$new(id = "credit",      # Name or label for the task
                       backend = credit,   # Dataframe
                       target = "risk")    # Target variable


#----- train model ----
# install.packages("ranger")
# libray(ranger)
# Goal here: Train a random forest on this dataset to predict credit risk
# and give me probabilities instead of hard labels.

# Implement a  random forest with soft output -> probabilities
mod = lrn("classif.ranger", predict_type = "prob")
set.seed(12005L)

# Train the model
mod$train(task)

#----- create predictor object ----
pred = Predictor$new(model = mod,
                     data = credit,
                     y = "risk",
                     type = "classification",
                     class = "good" # Focus on the probability of predicting the class ‘good’.
                     )

# what’s the prediction (probability of ‘good’ credit) for this specific configuration (x_interest)
pred$predict(x_interest)

# ----------------------------------------------------------------------------
# Info on PRIM: Patient Rule Induction Method
# ----------------------------------------------------------------------------
# It’s a way of finding a “box” (i.e., a hyperrectangle)
# where the prediction stays in a certain range (here, prob. from 0.3 to 0.6)
# ----------------------------------------------------------------------------

#----- compute regional descriptor with PRIM ---

set.seed(12005L)
prim = Prim$new(predictor = pred) # Use the model

# Find the box from x_interest
system.time({primb = prim$find_box(x_interest, desired_range = c(0.3, 0.6))})

# This gives you information on:
# - How big the box is (coverage)
# Whether all points inside it actually fall in the [0.3, 0.6] range (precision),
# Whether it contains x_interest (locality)
primb$evaluate()

# Impurity 0.24, dist 0.13
# Interpretation: Inside the box, 24% of the points do NOT fall within our
# desired prediction range (i.e., [0.3, 0.6] for class 'good')
# The reference point is pretty close to the middle of the box

# Visualize the “safe zone” in that 2D space
primb$plot_surface(feature_names = c("duration", "credit.amount"),
                   surface = "range")

#pp = PostProcessing$new(predictor = pred)
postproc = PostProcessing$new(predictor = pred)
postprocb = postproc$find_box(x_interest = x_interest, desired_range = c(0.3, 0.6),
  box_init = primb$box)
postprocb$evaluate()
postprocb
postprocb$plot_surface(feature_names = c("duration", "credit.amount"), surface = "range")

#----- compute regional descriptor with Maxbox ---
set.seed(12005L)
mb = MaxBox$new(predictor = pred, quiet = FALSE, strategy = "traindata")
system.time({mbb = mb$find_box(x_interest, desired_range = c(0.3, 0.6))})
mbb$evaluate()
mbb$plot_surface(feature_names = c("duration", "credit.amount"), surface = "range")
postproc = PostProcessing$new(predictor = pred)
postprocb = postproc$find_box(x_interest = x_interest, desired_range = c(0.3, 0.6),
  box_init = mbb$box)
postprocb$plot_surface(feature_names = c("duration", "credit.amount"), surface = "range")

#----- compute regional descriptor with MAIRE ----
mair = Maire$new(predictor = pred, num_of_iterations = 100L,
  convergence = TRUE, quiet = FALSE, strategy = "traindata")
system.time({mairb = mair$find_box(x_interest = x_interest, desired_range = c(0.3, 0.6))})
mairb$evaluate()
mairb$plot_surface(feature_names = c("duration", "credit.amount"), surface = "range")

#---- postprocess maire's box -----
postproc = PostProcessing$new(predictor = pred, subbox_relsize = 0.1)
postprocb = postproc$find_box(x_interest = x_interest, desired_range = c(0, 0.5), box_init = mairb$box)
postprocb$evaluate()
postprocb$plot_surface(feature_names = c("duration", "credit.amount"), surface = "range")

#---- compute regional descriptor with Anchors ----
anch = Anchor$new(predictor = pred)
system.time({box = anch$find_box(x_interest = x_interest,
  desired_range = c(0, 0.5))})
box$evaluate()
box$plot_surface(feature_names = c("duration", "credit.amount"), surface = "range")
