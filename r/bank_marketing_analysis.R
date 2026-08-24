# Bank Marketing Prediction
# Predicting term-deposit subscription using Naive Bayes, Logistic Regression and Random Forest

library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(ROSE)

# Load data
bank <- read.csv(file.path("data", "bank-full.csv"), sep = ";")

cat("Dataset dimensions:", nrow(bank), "rows x", ncol(bank), "columns\n")
print(head(bank))

# Replace "unknown" values with missing values
factor_cols <- c(
  "job", "marital", "education", "default",
  "housing", "loan", "contact", "month", "poutcome"
)

for (col in factor_cols) {
  bank[[col]] <- as.character(bank[[col]])
}

bank[bank == "unknown"] <- NA

# Remove incomplete rows
bank <- na.omit(bank)

cat("\nMissing values after cleaning:", sum(is.na(bank)), "\n")

# Remove any column containing only one unique value
unique_counts <- sapply(bank, function(x) length(unique(x)))
cols_to_remove <- names(unique_counts[unique_counts == 1])

if (length(cols_to_remove) > 0) {
  bank <- bank[, !names(bank) %in% cols_to_remove]
}

# Call duration is only known after the marketing call and can leak outcome information
bank$duration <- NULL

# Remove outliers from numeric columns using the IQR rule
num_cols <- names(bank)[sapply(bank, is.numeric)]

for (col in num_cols) {
  q1 <- quantile(bank[[col]], 0.25, na.rm = TRUE)
  q3 <- quantile(bank[[col]], 0.75, na.rm = TRUE)
  iqr_value <- q3 - q1

  lower_bound <- q1 - 1.5 * iqr_value
  upper_bound <- q3 + 1.5 * iqr_value

  bank <- bank[
    bank[[col]] >= lower_bound & bank[[col]] <= upper_bound,
  ]
}

# Convert categorical variables to factors
for (col in factor_cols) {
  if (col %in% names(bank)) {
    bank[[col]] <- factor(bank[[col]])
  }
}

bank$y <- factor(bank$y, levels = c("no", "yes"))

# Scale numeric variables
num_cols <- names(bank)[sapply(bank, is.numeric)]

for (col in num_cols) {
  bank[[col]] <- as.numeric(scale(bank[[col]]))
}

cat("\nClass distribution before balancing:\n")
print(prop.table(table(bank$y)))

# Split before oversampling so the test set remains untouched
set.seed(123)

train_index <- createDataPartition(bank$y, p = 0.8, list = FALSE)
train <- bank[train_index, ]
test <- bank[-train_index, ]

cat("\nTraining class distribution before balancing:\n")
print(prop.table(table(train$y)))

cat("\nTest class distribution:\n")
print(prop.table(table(test$y)))

# Balance only the training data
set.seed(123)

train_balanced <- ovun.sample(
  y ~ .,
  data = train,
  method = "over",
  N = 2 * max(table(train$y))
)$data

train_balanced$y <- factor(train_balanced$y, levels = levels(bank$y))

cat("\nTraining class distribution after balancing:\n")
print(prop.table(table(train_balanced$y)))

# Exploratory analysis
age_plot <- ggplot(train_balanced, aes(x = y, y = age, fill = y)) +
  geom_boxplot() +
  labs(
    title = "Age Distribution by Subscription",
    x = "Subscription",
    y = "Age"
  ) +
  theme_minimal()

print(age_plot)

scatter_plot <- ggplot(
  train_balanced,
  aes(x = age, y = balance, color = y)
) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Age vs Balance",
    x = "Age",
    y = "Balance"
  ) +
  theme_minimal()

print(scatter_plot)

# Train models
nb_model <- naiveBayes(y ~ ., data = train_balanced)

logit_model <- glm(
  y ~ .,
  data = train_balanced,
  family = "binomial"
)

set.seed(123)

rf_model <- randomForest(
  y ~ .,
  data = train_balanced,
  ntree = 100
)

# Confusion matrix plotting function
plot_confusion_matrix <- function(cm, model_name) {
  ggplot(as.data.frame(cm$table), aes(Reference, Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq)) +
    scale_fill_gradient(low = "white", high = "lightblue") +
    labs(
      title = paste("Confusion Matrix -", model_name),
      x = "Actual",
      y = "Predicted"
    ) +
    theme_minimal()
}

# Naive Bayes
nb_pred <- predict(nb_model, test)

nb_cm <- confusionMatrix(
  nb_pred,
  test$y,
  positive = "yes"
)

print(nb_cm)
print(plot_confusion_matrix(nb_cm, "Naive Bayes"))

# Logistic Regression
logit_prob <- predict(
  logit_model,
  test,
  type = "response"
)

logit_pred <- factor(
  ifelse(logit_prob > 0.5, "yes", "no"),
  levels = levels(test$y)
)

logit_cm <- confusionMatrix(
  logit_pred,
  test$y,
  positive = "yes"
)

print(logit_cm)
print(plot_confusion_matrix(logit_cm, "Logistic Regression"))

# Random Forest
rf_pred <- predict(rf_model, test)

rf_cm <- confusionMatrix(
  rf_pred,
  test$y,
  positive = "yes"
)

print(rf_cm)
print(plot_confusion_matrix(rf_cm, "Random Forest"))

# ROC objects and AUC
roc_nb <- roc(
  test$y,
  predict(nb_model, test, type = "raw")[, "yes"],
  levels = c("no", "yes"),
  direction = "<"
)

roc_logit <- roc(
  test$y,
  logit_prob,
  levels = c("no", "yes"),
  direction = "<"
)

roc_rf <- roc(
  test$y,
  predict(rf_model, test, type = "prob")[, "yes"],
  levels = c("no", "yes"),
  direction = "<"
)

# Model comparison
metrics <- data.frame(
  Model = c("Naive Bayes", "Logistic Regression", "Random Forest"),
  Accuracy = c(
    unname(nb_cm$overall["Accuracy"]),
    unname(logit_cm$overall["Accuracy"]),
    unname(rf_cm$overall["Accuracy"])
  ),
  Precision = c(
    unname(nb_cm$byClass["Precision"]),
    unname(logit_cm$byClass["Precision"]),
    unname(rf_cm$byClass["Precision"])
  ),
  Recall = c(
    unname(nb_cm$byClass["Recall"]),
    unname(logit_cm$byClass["Recall"]),
    unname(rf_cm$byClass["Recall"])
  ),
  F1 = c(
    unname(nb_cm$byClass["F1"]),
    unname(logit_cm$byClass["F1"]),
    unname(rf_cm$byClass["F1"])
  ),
  AUC = c(
    as.numeric(auc(roc_nb)),
    as.numeric(auc(roc_logit)),
    as.numeric(auc(roc_rf))
  )
)

print(metrics)

# ROC curve comparison
plot(
  roc_nb,
  col = "blue",
  main = "ROC Curves Comparison"
)

lines(roc_logit, col = "red")
lines(roc_rf, col = "green")

legend(
  "bottomright",
  legend = c(
    sprintf("Naive Bayes (AUC = %.2f)", auc(roc_nb)),
    sprintf("Logistic Regression (AUC = %.2f)", auc(roc_logit)),
    sprintf("Random Forest (AUC = %.2f)", auc(roc_rf))
  ),
  col = c("blue", "red", "green"),
  lwd = 2
)

# Random Forest feature importance
varImpPlot(
  rf_model,
  main = "Random Forest Feature Importance",
  color = "steelblue",
  pch = 19
)
