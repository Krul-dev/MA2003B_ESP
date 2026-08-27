library(tidyverse)
library(MASS)
library(here)

# Set seed so that the generated dataset can be reproduced
set.seed(123)

# ------------------------------------------------------------
# 1. Define the population parameters
# ------------------------------------------------------------

# Number of observations
n <- 600

# Prior probabilities for the six groups
priors <- c(
  A = 0.12,
  B = 0.18,
  C = 0.15,
  D = 0.20,
  E = 0.10,
  F = 0.25
)

# Mean vector for each group
#
# X1, X2, X3, and X4 contain discriminant information:
# their population means depend on the group.
#
# X5, X6, X7, and X8 are noise variables:
# their population means are zero in every group.

means <- list(
  A = c(-2.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  B = c(2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
  C = c(0.0, 2.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0),
  D = c(0.0, 0.0, -2.0, -1.5, 0.0, 0.0, 0.0, 0.0),
  E = c(0.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0),
  F = c(-0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)

# Marginal standard deviations of the eight variables
sigma <- c(
  1.0,
  1.3,
  0.9,
  1.5,
  1.1,
  1.4,
  0.8,
  1.2
)

# ------------------------------------------------------------
# 2. Construct the common covariance matrix
# ------------------------------------------------------------

# The first four variables form the signal block.
# The last four variables form the noise block.
#
# Within each block, correlations decrease as variables
# become farther apart:
#
# cor(X_i, X_j) = rho^|i - j|
#
# The two blocks are independent of each other.
#
# This ensures that X5, ..., X8 do not contain information
# about the group, either directly through their means or
# indirectly through correlation with the signal variables.
#
# Every group has the same covariance matrix, as assumed by LDA.

rho <- 0.45

# Correlation matrix for a block of four variables
correlation_block <- outer(
  1:4,
  1:4,
  function(i, j) rho^abs(i - j)
)

# Covariance matrix for the signal variables X1, ..., X4
D_signal <- diag(sigma[1:4])

Sigma_signal <- D_signal %*%
  correlation_block %*%
  D_signal

# Covariance matrix for the noise variables X5, ..., X8
D_noise <- diag(sigma[5:8])

Sigma_noise <- D_noise %*%
  correlation_block %*%
  D_noise

# Construct the complete covariance matrix
#
# Sigma = [ Sigma_signal      0       ]
#         [      0       Sigma_noise ]

Sigma <- matrix(
  0,
  nrow = 8,
  ncol = 8
)

Sigma[1:4, 1:4] <- Sigma_signal
Sigma[5:8, 5:8] <- Sigma_noise

# Add variable names for easier inspection
rownames(Sigma) <- paste0("X", 1:8)
colnames(Sigma) <- paste0("X", 1:8)

Sigma

# ------------------------------------------------------------
# 3. Generate the group memberships
# ------------------------------------------------------------

# Each observation is assigned to one of the six groups
# according to the specified prior probabilities.

group <- sample(
  names(priors),
  size = n,
  replace = TRUE,
  prob = priors
)

# ------------------------------------------------------------
# 4. Generate the eight explanatory variables
# ------------------------------------------------------------

# For an observation belonging to group g, generate
#
# X | G = g ~ N_8(mu_g, Sigma)
#
# The first four variables contain group information.
# The last four variables have the same distribution
# in every group and therefore act as noise predictors.

X <- map_dfr(
  group,
  \(g) {
    x <- mvrnorm(
      n = 1,
      mu = means[[g]],
      Sigma = Sigma
    )

    names(x) <- paste0("X", 1:8)

    as_tibble_row(x)
  }
)

# ------------------------------------------------------------
# 5. Construct the final dataset
# ------------------------------------------------------------

lda_data <- X |>
  mutate(
    group = factor(
      group,
      levels = names(priors)
    )
  ) |>
  relocate(group)

# ------------------------------------------------------------
# 6. Inspect the generated dataset
# ------------------------------------------------------------

print(lda_data)

# Check dimensions:
# 600 observations, 8 explanatory variables, and 1 group variable
print(dim(lda_data))

# Number of observations in each group
group_counts <- lda_data |>
  count(group)

print(group_counts)

# Sample proportions of the groups
group_proportions <- lda_data |>
  count(group) |>
  mutate(
    proportion = n / sum(n)
  )

print(group_proportions)

# Sample means for each group
#
# We expect:
#
# - X1, X2, X3, and X4 to show differences between groups.
# - X5, X6, X7, and X8 to have sample means close to zero
#   in every group.

group_means <- lda_data |>
  group_by(group) |>
  summarise(
    across(
      starts_with("X"),
      mean
    ),
    .groups = "drop"
  )

print(group_means)

# ------------------------------------------------------------
# 7. Verify the signal/noise structure
# ------------------------------------------------------------

# Run a separate one-way ANOVA for each explanatory variable.
#
# For each X_j:
#
# H0: the population mean of X_j is the same in all groups.
#
# We expect X1, ..., X4 to reject H0 and
# X5, ..., X8 not to reject H0.

anova_results <- lda_data |>
  pivot_longer(
    cols = starts_with("X"),
    names_to = "variable",
    values_to = "value"
  ) |>
  group_by(variable) |>
  summarise(
    model = list(aov(value ~ group)),
    .groups = "drop"
  ) |>
  mutate(
    anova = map(model, broom::tidy)
  ) |>
  unnest(anova) |>
  filter(term == "group") |>
  dplyr::select(
    variable,
    statistic,
    p.value
  )

print(anova_results)

# ------------------------------------------------------------
# 9. Save the dataset as a CSV file
# ------------------------------------------------------------

write_csv(
  lda_data,
  here("data", "linear_discriminant_selection_data.csv")
)
