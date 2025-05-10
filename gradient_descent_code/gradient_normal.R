normal_loss <- function(x, theta){
  mu <- theta[1]
  sigma2 <- theta[2]
  n <- length(x)
  
  # Prevent log of zero or negative variance
  if (sigma2 <= 0) return(Inf)
  
  loss <- (n/2)*log(sigma2) + sum((x - mu)^2) / (2*sigma2)
  return(loss)
}

normal_grad <- function(x, theta){
  mu <- theta[1]
  sigma2 <- theta[2]
  n <- length(x)
  
  # Prevent division by zero or negative variance
  if (sigma2 <= 0) return(c(NA, NA))
  
  d_mu <- -sum(x - mu) / sigma2
  d_sigma2 <- (n / (2 * sigma2)) - (sum((x - mu)^2) / (2 * sigma2^2))
  return(c(d_mu, d_sigma2))
}

gradient_descent <- function(x, theta_init, step_size, threshold, max_iter=10000) {
  theta <- theta_init
  loss_old <- normal_loss(x, theta)
  iterations <- 0
  loss_history <- c(loss_old)  # Store the first loss
  
  repeat {
    grad <- normal_grad(x, theta)
    
    # Stop if invalid gradient (from bad variance)
    if (any(is.na(grad))) {
      stop("Gradient returned NA due to invalid variance.")
    }
    
    theta_new <- theta - step_size * grad
    
    # Prevent negative variance
    if (theta_new[2] <= 0) theta_new[2] <- 1e-6
    
    loss_new <- normal_loss(x, theta_new)
    loss_history <- c(loss_history, loss_new)  # Append new loss
    
    if (abs(loss_new - loss_old) < threshold || iterations >= max_iter) {
      break
    }
    
    theta <- theta_new
    loss_old <- loss_new
    iterations <- iterations + 1
  }
  
  return(list(beta = theta, iterations = iterations, loss_history = loss_history))
}

# Load data
data <- read.csv("C:\\Users\\Harsh\\Downloads\\synthetic data - problem-4.csv")
x_data <- data$x


