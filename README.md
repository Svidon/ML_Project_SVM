# ML_Project_SVM
---

This project aims at using an SVM (in this case with `rbf` kernel) to predict if the email described by the data given (more infos in `spambase.names` and `spambase.DOCUMENTATION`) are spam mails or not.
The process involves performing a GridSearch on `gamma` and `C` parameters of the SVM classifier, using a 3Fold cross-validation.
After that the learning curve of the algorithm with optimal parameters is plotted (`RBF_learning_curve.png`).
At last the model is trained over the full training set and testing it on the test set. The final accuracy is **0.93**.
