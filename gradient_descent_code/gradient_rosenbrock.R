
rosenbrock_loss <- function(x){
  loss <- (1-x[1])^2 + 100*(x[2]-(x[1])^2)^2
  return(loss)
}
rosenbrock_grad <- function(x){
  grad1 <- -2*(1-x[1]) - 400*x[1]*(x[2]-(x[1])^2)
  grad2 <- 200*(x[2]- (x[1])^2)
  grad <- rbind(grad1,grad2)
  return(as.vector(grad))
}
gradient_descent <- function(beta_0, step_size, threshold) {
  converged <- FALSE
  iterations <- 0
  beta <- beta_0
  loss_old <- rosenbrock_loss(beta)
  loss_history <- c(loss_old)  # Store the first loss
  
  
  while (!converged) {
    gradient <- rosenbrock_grad(beta)  # Compute gradient
    beta_new <- beta - step_size * gradient  # Update beta using gradient and step size
    loss_new <- rosenbrock_loss(beta_new)  # Compute new loss
    loss_history <- c(loss_history, loss_new)  # Append new loss
    
    
    # Check for convergence
    if (abs(loss_new - loss_old) < threshold) {
      converged <- TRUE
    }
    
    beta <- beta_new  # Update beta for the next iteration
    loss_old <- loss_new  # Update old loss for next iteration
    iterations <- iterations + 1
  }
  
  return(list(beta = beta, iterations = iterations, loss_history = loss_history))
}
result <- gradient_descent(c(1, -1), 0.001, 1e-6)

# Plot
plot(result$loss_history, type = "l", col = "blue",
     xlab = "Iteration", ylab = "Loss",
     main = "Loss vs Iterations (Rosenbrock Function)")

