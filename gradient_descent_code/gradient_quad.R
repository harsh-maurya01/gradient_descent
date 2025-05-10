
quad_loss <- function(x){
  loss <- 2*(x[1])^2 + 4*(x[2])^2 - 4*(x[1]) - 8*(x[2])
  return (loss)
}
quad_grad <- function(x){
  grad <- c((4*x[1] -4), (8*x[2] - 8))
  return (as.vector(grad))
}

gradient_descent <- function(beta_0, step_size, threshold, max_iter = 10000) {
  converged <- FALSE
  iterations <- 0
  beta <- beta_0
  loss_old <- quad_loss(beta)
  loss_history <- c(loss_old)  # Initialize with first loss
  
  while (!converged && iterations < max_iter) {
    gradient <- quad_grad(beta)
    beta_new <- beta - step_size * gradient
    loss_new <- quad_loss(beta_new)
    
    loss_history <- c(loss_history, loss_new)
    
    if (abs(loss_new - loss_old) < threshold) {
      converged <- TRUE
    }
    
    beta <- beta_new
    loss_old <- loss_new
    iterations <- iterations + 1
  }
  
  return(list(beta = beta, iterations = iterations, loss_history = loss_history))
}

result <- gradient_descent(c(0, 0), 0.1, 1e-6)

plot(result$loss_history, type = "l", col = "red",
     xlab = "Iteration", ylab = "Loss",
     main = "Loss vs Iterations (Quadratic Function)")
