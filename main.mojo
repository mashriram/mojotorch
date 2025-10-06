from mojotorch import MojTensor, randn, Linear, Sequential, ReLU, Sigmoid, Identity, Adam, bce_loss

fn main():
    print("MojoTorch v3 - A Deep Learning Library")
    print("---------------------------------------")
    print("\nTraining a Binary Classifier...")

    var model = Sequential(
        Linear(2, 16),
        Sequential(
            ReLU(),
            Sequential(
                Linear(16, 1),
                Sequential(Sigmoid(), Identity()))))
    

    var n_samples = 100
    var X_train = randn(List(n_samples, 2))
    var y_true = MojTensor(List(n_samples, 1))
    for i in range(n_samples):
        var x1 = X_train[i * 2 + 0]
        var x2 = X_train[i * 2 + 1]
        y_true[i] = 1.0 if (x1 + x2) > 0.5 else 0.0

    var optimizer = Adam(model.parameters(), lr=0.01)
    var epochs = 25
    print("Using Adam Optimizer and BCELoss.")
    for i in range(epochs):
        var y_pred = model.forward(X_train)
        var loss = bce_loss(y_pred, y_true)

        if (i + 1) % 5 == 0:
            print("Epoch", i + 1, "| Loss:", loss[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    var y_final = model.forward(X_train)
    var correct_predictions = 0
    for i in range(n_samples):
        var pred_val = y_final[i]
        var true_val = y_true[i]
        if (pred_val > 0.5 and true_val == 1.0) or (pred_val <= 0.5 and true_val == 0.0):
            correct_predictions += 1

    var accuracy = Float64(correct_predictions) / Float64(n_samples) * 100.0
    print("Training finished.")
    print("Final Accuracy:", accuracy, "%")
