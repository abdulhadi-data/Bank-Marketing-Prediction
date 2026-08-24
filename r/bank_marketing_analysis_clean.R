# Load required libraries
library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(ROSE)

# Load the data
bank <- read.csv(file.path("data", "bank-full.csv"), sep = ";")
cat("Head of dataset:\n")
print(head(bank))

# Check missing values
cat("\nMissing values count (before cleaning):\n")
print(sum(is.na(bank)))

# Replace "unknown" with NA
factor_cols <- c("job", "marital", "education", "default", 
                 "housing", "loan", "contact", "month", "poutcome")

for (col in factor_cols) {
  bank[[col]] <- as.character(bank[[col]])
}
bank[bank == "unknown"] <- NA

# Remove rows with NA
bank <- na.omit(bank)
cat("\nMissing values count (after cleaning):\n")
print(sum(is.na(bank)))

# Show summary
cat("\nSummary of dataset:\n")
print(summary(bank))

# Check for columns with only one unique value and remove them

unique_counts <- c()  # Empty vector to store counts

# Loop through each column and count unique values
for (col_name in names(bank)) {
  unique_counts[col_name] <- length(unique(bank[[col_name]]))
}

print("\nUnique values per column:")
print(unique_counts)

# Find columns with only one unique value
cols_to_remove <- names(unique_counts[unique_counts == 1])

# Remove columns if any need to be removed
if (length(cols_to_remove) > 0) {
  bank <- bank[, !names(bank) %in% cols_to_remove]
}

# Remove the duration column as it's not useful and may leak information
bank$duration <- NULL

# Remove outliers from numeric columns using the IQR method
num_cols <- names(bank)[sapply(bank, is.numeric)]
for(col in num_cols){
  Q1 <- quantile(bank[[col]], 0.25)
  Q3 <- quantile(bank[[col]], 0.75)
  IQR_value <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  
  bank <- bank[bank[[col]] >= lower_bound & bank[[col]] <= upper_bound, ]
}

 
# Convert categorical columns to factors
for(col in factor_cols){
  bank[[col]] <- factor(bank[[col]])
}

# Convert target variable 'y' to factor
bank$y <- factor(bank$y)

# Scale numeric columns after outlier removal
for(col in num_cols){
  bank[[col]] <- scale(bank[[col]])
}

# Check original class distribution
cat("\nOriginal class distribution:\n")
print(table(bank$y) / length(bank$y))

# Perform random oversampling to balance the target classes
set.seed(123)

# Use ROSE to oversample the minority class

# Set total observations to twice the largest class count
bank_balanced <- ovun.sample(y ~ ., data = bank, method = "over", N = 2 * max(table(bank$y)))$data
cat("\nBalanced class distribution:\n")
print(prop.table(table(bank_balanced$y)))

 
#    DATA ANALYSIS & VISUALIZATION  #


# Age distribution by subscription
age_plot <- ggplot(bank_balanced, aes(x = y, y = age, fill = y)) +
  geom_boxplot() +
  labs(title = "Age Distribution by Subscription",
       x = "Subscription",
       y = "Age") +
  theme_minimal()
print(age_plot)

# Age vs balance
scatter_plot <- ggplot(bank_balanced, aes(x = age, y = balance, color = y)) +
  geom_point() +
  labs(title = "Age vs Balance",
       x = "Age",
       y = "Balance") +
  theme_minimal()

print(scatter_plot)


#    DATA SPLITTING          #

 
set.seed(123)
train_index <- createDataPartition(bank_balanced$y, p = 0.8, list = FALSE)
train <- bank_balanced[train_index, ]
test <- bank_balanced[-train_index, ]

cat("\nTraining set distribution:\n")
print(table(train$y) / nrow(train))

# Train models
# 1. Naive Bayes
nb_model <- naiveBayes(y ~ ., data = train)

# 2. Logistic Regression
logit_model <- glm(y ~ ., data = train, family = "binomial")

# 3. Random Forest
set.seed(123)
rf_model <- randomForest(y ~ ., data = train, ntree = 100)

 
# Evaluate models
 
plot_confusion_matrix <- function(cm, model_name) {
  ggplot(as.data.frame(cm$table), aes(Reference, Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq)) +
    scale_fill_gradient(low = "white", high = "lightblue") +  # lighter colour
    labs(title = paste("Confusion Matrix -", model_name),
         x = "Actual", y = "Predicted")

}

# Naive Bayes Evaluation
nb_pred <- predict(nb_model, test)
nb_cm <- confusionMatrix(nb_pred, test$y, positive = "yes")
print(nb_cm)  # prints numeric confusion matrix
print(plot_confusion_matrix(nb_cm, "Naive Bayes"))  # display plot

# Logistic Regression Evaluation
logit_prob <- predict(logit_model, test, type = "response")
logit_pred <- factor(ifelse(logit_prob > 0.5, "yes", "no"), levels = levels(test$y))
logit_cm <- confusionMatrix(logit_pred, test$y, positive = "yes")
print(plot_confusion_matrix(logit_cm, "Logistic Regression"))

# Random Forest Evaluation
rf_pred <- predict(rf_model, test)
rf_cm <- confusionMatrix(rf_pred, test$y, positive = "yes")
print(rf_cm)  # prints numeric confusion matrix
print(plot_confusion_matrix(rf_cm, "Random Forest"))  # display plot

# Metrics Comparison

metrics <- data.frame(
  Model = c("Naive Bayes", "Logistic Regression", "Random Forest"),
  Accuracy = c(nb_cm$overall["Accuracy"],
               logit_cm$overall["Accuracy"],
               rf_cm$overall["Accuracy"]),
  
  Precision = c(nb_cm$byClass["Precision"],
                logit_cm$byClass["Precision"],
                rf_cm$byClass["Precision"]),
  
  Recall = c(nb_cm$byClass["Recall"],
             logit_cm$byClass["Recall"],
             rf_cm$byClass["Recall"]),
  
  F1 = c(nb_cm$byClass["F1"],
         logit_cm$byClass["F1"],
         rf_cm$byClass["F1"]),
  
  AUC = c(auc(roc(test$y, predict(nb_model, test, type = "raw")[,"yes"])),
          auc(roc(test$y, predict(logit_model, test, type = "response"))),
          auc(roc(test$y, predict(rf_model, test, type = "prob")[,"yes"])))
)

print(metrics)

 
# ROC Curves Comparison

roc_nb <- roc(test$y, predict(nb_model, test, type = "raw")[,"yes"])
roc_logit <- roc(test$y, predict(logit_model, test, type = "response"))
roc_rf <- roc(test$y, predict(rf_model, test, type = "prob")[,"yes"])

plot(roc_nb, col = "blue", main = "ROC Curves Comparison")
lines(roc_logit, col = "red")
lines(roc_rf, col = "green")

legend("bottomright",
       legend = c(sprintf("Naive Bayes (AUC = %.2f)", auc(roc_nb)),
                  sprintf("Logistic Regression (AUC = %.2f)", auc(roc_logit)),
                  sprintf("Random Forest (AUC = %.2f)", auc(roc_rf))),
       col = c("blue", "red", "green"),
       lwd = 2)

 
# Feature importance
varImpPlot(rf_model, main = "Random Forest Feature Importance", 
           color = "steelblue", pch = 19)
