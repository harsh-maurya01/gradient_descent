# Data setup
n = 100
x_vec <- rnorm(n, 1, sqrt(2))  
y_vec <- rnorm(n, 2 + 3 * x_vec, sqrt(5))  
# Loss function

loss <- function(x_raw, y, beta) {
  n <- length(y)  
  X_mat <- rbind(rep(1, n), x_raw) 
  X_mat <- t(X_mat) 
  prediction <- X_mat %*% beta 
  residuals <- y - prediction  
  loss_value <- (1/(2*n)) * sum(residuals^2)  # Loss function
  return(loss_value)
}

# Gradient function
grad <- function(x_raw, y, beta) {
  n <- length(y)  
  X_mat <- rbind(rep(1, n), x_raw)  
  X_mat <- t(X_mat)  
  prediction <- X_mat %*% beta 
  residuals <- y - prediction  
  grad_value <- - (1/n) * (t(X_mat) %*% residuals)
  return(grad_value)
}

# Gradient descent function
gradient_descent <- function(x_raw, y, beta_0, grad, step_size, threshold) {
  converged <- FALSE
  iterations <- 0
  beta <- beta_0
  loss_old <- loss(x_raw, y, beta)
  loss_history <- c(loss_old)
  beta_history <- matrix(beta_0, nrow = 1)  # Store each beta as a row
  
  while (!converged) {
    gradient <- grad(x_raw, y, beta)
    beta_new <- beta - step_size * gradient
    loss_new <- loss(x_raw, y, beta_new)
    
    loss_history <- c(loss_history, loss_new)
    beta_history <- rbind(beta_history, as.vector(beta_new))
    
    if (abs(loss_new - loss_old) < threshold) {
      converged <- TRUE
    }
    
    beta <- beta_new
    loss_old <- loss_new
    iterations <- iterations + 1
  }
  
  return(list(beta = beta, iterations = iterations,
              loss_history = loss_history,
              beta_history = beta_history))
}

# Run gradient descent
result <- gradient_descent(x_vec, y_vec, c(0, 0), grad, 0.01, 1e-6)


# Extract beta history
beta_mat <- result$beta_history
final_beta <- result$beta  # final learned parameters



