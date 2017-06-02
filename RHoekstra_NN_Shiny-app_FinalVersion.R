########################
## Neural Network App ##
## XOR-Function       ##
## Versie 5           ##
## Ria Hoekstra       ##
## 02/06/2017         ##
########################
rm(list = ls())

library(shiny)
library(shinythemes)

ui <- fluidPage(theme = shinytheme("flatly"),
  titlePanel("Neural Network Tutorial"),
  tabsetPanel(
    tabPanel("Training",
             
             sidebarLayout(
               
               sidebarPanel(
                 
                 helpText("Move the slider from left to right to pick the number of iterations"),
                 sliderInput(inputId = "num", label = "Choose a number",
                             value = 1000, min = 1000, max = 10000, step = 1000),
                 br(),
                 helpText("Move the slider from left to right to choose a learning rate"),
                 sliderInput(inputId = "learn_rate", label = "Choose a learning rate",
                             value = 0.8, min = 0, max = 1, step = 0.02),
                 br(),
                 helpText("Click save to save the weights from the Neural Network"),
                 actionButton(inputId = "save", label = "SAVE")
                 
               ),
               
               mainPanel(
                 
                 h3("Training the Neural Network"), 
                 br(),
                 p("This Shiny app will introduce you to Artificial Neural Networks (ANN). 
                   ANN consist of neurons and synapses. 
                   Neurons are the nodes in the network and synapses the connections. 
                   The general idea of ANN is to train the model trough forward and back propagation so the model can find the optimal weights.  
                   For more information on neural networks click on the information tap."),
                 br(),
                 p("For this introduction a XOR-function is modeled. 
                   A XOR-function takes 4 sets of input values: 0,0 0,1 1,0 1,1 with its associated targets 0, 1, 1, 0. 
                   You can train the model yourself by playing around with the number of iterations, and the learning rate. 
                   The plot bellow shows you how well the model is trained.  
                   Once you are satisfied with the training of the model you can click on the save button and move on to the next tap to test the model and see for yourself whether it can predict the target outcome accurately."),
                 plotOutput(outputId = "Error")
                 
                 )
               )
             ),
    
    tabPanel("Testing",
             sidebarLayout(
               
               sidebarPanel(
                 
                 helpText("Choose one of the input values to test the model"),
                 radioButtons("choise", label = "Input Values",
                              choices = list("0, 0" = 1, "0, 1" = 2,
                                             "1, 0" = 3, "1, 1" = 4),selected = 1),
                 br(),
                 helpText("Click test to test the input for the Neural Network"),
                 actionButton(inputId = "test", label = "TEST")
               ),
               
               mainPanel(

                 h3("Testing the Neural Network"),
                 br(),
                 p("If you have trained the model in the training tab, and or satisfied with the performance of the neural network, you can now test the neural network.
                   From the left panel you can choose an input value. 
                   Below the corresponding target and the calculated value will be shown, so you can see whether for this specific input value, the neural network is well trained."),
                 textOutput(outputId = "show_test1"),
                 textOutput(outputId = "show_test2"),
                 textOutput(outputId = "show_test3")
               )
             )
    ),
    
    tabPanel("Information",
             mainPanel(
               
               h3("Introducing Neural Networks"),
               br(),
               p("Artificial Neural Networks (ANN) are statistical learning models that are used in machine learning. 
                  The network consists of a system of “neurons” that are connected trough “synapses” which can send messages to one another. 
                  The connections within the network can be systematically adjusted based on input and output, which makes these kinds of models ideal for supervised learning. "),
               p("A Neural network consists of three main parts: an input layer, a hidden layer and an output layer, see figure 1. 
                  Although you can have multiple hidden layers (“deep” learning), for this tutorial we will just stick with one hidden layer, as multiple hidden layers are only useful for more complex models. 
                  These layers are “hidden” since they are not visible as the network input or output."),
               p("In a Neural network circles represent neurons and lines represent synapses. 
                  Synapses have a relatively easy job. 
                  They take a number from their input value and multiply it with a specific weight. 
                  Neurons are a bit more complicated. 
                  They have to add the value for al their input and apply an activation function.  
                  For this tutorial the sigmoid activation function is used."),
               p("Training a neural network means finding the optimal weights by repeating two key steps, namely forward propagation and back propagation. 
                  In the first forward propagation a set of randomly selected weights between 0 and 1 is applied to the input data to calculate an output. 
                  In back propagation the margin of error of the output is measured to adjust the weights accordingly. 
                  These new adjusted weights are then used for the next forward propagation. 
                  Neural networks repeat both forward and back propagation until the weights are optimized (i.e. can accurately predict an output).")
                      )
             )
    
   )
)

server <- function(input, output){
  
  ##########################################
  ## Function to Train the Neural Network ##
  ##########################################
  func_train <- observe({
    
    calculated <<- c()
    targets    <<- c()
    learn_rate <<- input$learn_rate
    # This line of code creates the bug when using the learning rate slider
    # learn_rate <- input$learn_rate 
    
    for (i in 1:input$num) {
      
      ##################
      ## Sample input ##
      ##################
      data <- list(c(0,0), c(0,1), c(1,0), c(1,1))    
      input <- sample(data, 1)
      
      # Determine the target
      if (input[[1]][1] == 0 & input[[1]][2] == 0){
        target <- 0
      } else if (input[[1]][1] == 0 & input[[1]][2] == 1){
        target <- 1
      } else if (input[[1]][1] == 1 & input[[1]][2] == 0){
        target <- 1
      } else {
        target <- 0
      }
      targets[i] <<- target
      
      #########################
      ## Forward Propagation ##
      #########################
      if (i == 1) {
        
        # For the first forward propagation sample weights between 0 and 1
        first_weights <- matrix(runif(6, 0, 1), 2, 3)               
        second_weights <- matrix(runif(3, 0, 1), 1, 3) 
        
      } else {
        
        # For the rest of forward propagation weights are result from back propagation
        first_weights  <- delta_second_weights
        second_weights <- delta_first_weights
      }
      
      # Calculate the hidden sum for each node
      hidden_layer_sum <- apply(input[[1]] * first_weights, 2, sum)     
      
      ######################
      ## sigmoid function ##
      ######################
      S <- function(x)
      {
        1/(1 + exp(-x))
      }
      
      # Calculate the hidden layer results for each node
      hidden_layer_result <- S(hidden_layer_sum)
      
      # Calculate the output sum for output node
      output_sum          <- sum(hidden_layer_result * second_weights)
      
      # Activate the output sum to get calculated output value for forward propagation 
      calculated[i]       <<- S(output_sum)
      
      ######################
      ## Back propagation ##
      ######################
      # Calculate the marging of error between target value and calculated value
      output_sum_margin_of_error <- targets[i] - calculated[i]
      
      ####################################
      ## derivative of sigmoid function ##
      ####################################
      derS <- function(x)
      {    
        exp(x)/((exp(x) + 1)^2)
      }
      
      # Calculate new node output sum 
      delta_output_sum      <- derS(output_sum) * output_sum_margin_of_error 
      
      # Calculate the hidden output changes 
      hidden_output_changes <- delta_output_sum * hidden_layer_result * learn_rate                        
      
      # Calculate new second weights for forward propagation
      delta_first_weights   <<- second_weights + hidden_output_changes
      
      # Calculate new node hidden sum
      delta_hidden_sum      <- delta_output_sum * second_weights * derS(hidden_layer_sum) * learn_rate
      
      # Calculate delta weights 
      delta_weights         <- c(delta_hidden_sum * input[[1]][1], delta_hidden_sum * input[[1]][2]) 
      
      # Calculate new first weights for forward propagation 
      delta_second_weights  <<- first_weights + delta_weights
      
    } 
  })
  
  ##############################
  ## Plot for Training values ##
  ##############################
  output$Error <- renderPlot({
    plot(targets, 
         xlim = c(0,input$num), 
         ylim = c(0,1), 
         xlab = "Number of iterations", 
         ylab = " ", 
         las  = 1, 
         frame.plot = FALSE, 
         pch  = 16, 
         col  = "green4") 
    points(calculated, pch = 16, col = "red")
  })
  
  ################################
  ## save weights from training ##
  ################################
  observeEvent(input$save, {
    saved_first_weights  <<- delta_second_weights
    saved_second_weights <<- delta_first_weights
  })
  
  ##################################################
  ## Function to test the trainend Neural Network ##
  ##################################################
  func_test <- observe({
    
    # Determine input
    if(input$choise == 1){
      test_input <<- c(0,0)
    } else if(input$choise == 2){
      test_input <<- c(0,1)
    } else if(input$choise == 3){
      test_input <<- c(1,0)
    } else {
      test_input <<- c(1,1)
    }
    
    # Determine test target
    if(input$choise == 1){
      test_target <<- 0
    } else if(input$choise == 2){
      test_target <<- 1
    } else if(input$choise == 3){
      test_target <<- 1
    } else {
      test_target <<- 0
    }
    
    # To test the model we only need to do forward propagation with our trained set of weights 
    hidden_layer_sum    <- apply(test_input * delta_second_weights, 2, sum)
  
    ######################
    ## sigmoid function ##
    ######################
    S <- function(x)
    {
      1/(1 + exp(-x))
    }
    
    hidden_layer_result <- S(hidden_layer_sum)                  
    output_sum          <- sum(hidden_layer_result * delta_first_weights)
    test_calculated     <<- S(output_sum)
  })
  
  ####################
  ## Testing output ##
  ####################
  output$show_test1 <- renderText({
    paste("The target =", test_target, "\n") 
  })
  
  output$show_test2 <- renderText({
    paste("The calculated value =", test_calculated, "\n") 
  })
  
  output$show_test3 <- renderText({
    paste("Do you think the network is well trainend for these input values?", "\n") 
  })
  
}

shinyApp(ui = ui, server = server)
