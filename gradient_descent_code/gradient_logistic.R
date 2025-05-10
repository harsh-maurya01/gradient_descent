
data <- read.csv("C:\\Users\\Harsh\\Downloads\\synthetic data - problem-2.csv")

x1 <- data$x1
x2 <- data$x2
y <- data$y



# Standard logistic regression loss function (negative log-likelihood)
loss <- function(x1, x2, y, beta) {
  X_matrix <- cbind(x1, x2)  # Create design matrix properly
  z <- X_matrix %*% beta     # Linear predictor
  
  # Log-likelihood calculation
  log_likelihood <- sum(y * z - log(1 + exp(z)))
  
  # Return negative log-likelihood (loss)
  return(-log_likelihood)
}

# Correct gradient calculation for logistic regression
grad <- function(x1, x2, y, beta) {
  X_matrix <- cbind(x1, x2)
  z <- X_matrix %*% beta
  
  # Calculate sigmoid probabilities
  p <- 1 / (1 + exp(-z))
  
  # Gradient is X^T * (y - p)
  gradient <- -t(X_matrix) %*% (y - p)
  
  return(as.vector(gradient))
}

# Gradient descent function looks mostly good
gradient_descent_vectorized <- function(x1, x2, y, beta_0, grad, loss, step_size, threshold) {
  converged <- FALSE
  iterations <- 0
  beta <- beta_0
  loss_old <- loss(x1, x2, y, beta)
  loss_history <- c(loss_old)  # Initialize with first loss
  
  
  while (!converged) {
    gradient <- grad(x1, x2, y, beta)
    beta_new <- beta - step_size * gradient
    loss_new <- loss(x1, x2, y, beta_new)
    loss_history <- c(loss_history, loss_new)
    
    # Check for convergence
    if (abs(loss_new - loss_old) < threshold) {
      converged <- TRUE
    }
    
    beta <- beta_new
    loss_old <- loss_new
    iterations <- iterations + 1
    
    # Add a safety check to prevent infinite loops
    if (iterations > 10000) {
      warning("Maximum iterations reached")
      break
    }
  }
  
  return(list(beta = beta, iterations = iterations, loss_history = loss_history))
}

# Initialize beta with zeros
beta_0 <- c(0, 0)

# Run gradient descent
result <- gradient_descent_vectorized(x1, x2, y, beta_0, grad, loss, 0.05, 1e-5)
