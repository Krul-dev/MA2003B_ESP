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
means <- list(
  A = c(-2.0, -1.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
  B = c(2.0, -1.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
  C = c(0.0, 2.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0),
  D = c(0.0, 0.0, -2.0, -1.5, 1.0, 0.0, 0.5, 0.0),
  E = c(0.5, 0.0, 0.0, 2.0, 1.5, 1.0, 0.0, 0.5),
  F = c(-0.5, 0.5, 0.0, 0.0, -1.5, -2.0, -0.5, -1.0)
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

# Correlations decrease as variables become farther apart:
#
# cor(X_i, X_j) = rho^|i - j|
#
# Every group has the same covariance matrix, as assumed by LDA.

rho <- 0.45

correlation_matrix <- outer(
  1:8,
  1:8,
  function(i, j) rho^abs(i - j)
)

# Convert the correlation matrix into a covariance matrix:
#
# Sigma = D R D
#
# where D contains the marginal standard deviations.

Sigma <- diag(sigma) %*%
  correlation_matrix %*%
  diag(sigma)

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
# mvrnorm(n = 1, ...) returns an unnamed numeric vector.
# We therefore assign variable names before converting it
# into a one-row tibble.

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
# 7. Create the data directory if necessary
# ------------------------------------------------------------

dir.create(
  here("data"),
  showWarnings = FALSE
)

# ------------------------------------------------------------
# 8. Save the dataset as a CSV file
# ------------------------------------------------------------

write_csv(
  lda_data,
  here("data", "linear_discriminant_data.csv")
)
