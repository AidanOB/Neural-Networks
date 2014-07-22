function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

    % Useful values
    m = size(X, 1);

    % Add bias to the input
    X = [ones(m, 1) X];
    
    %Perform the first layer of the NN
    a1 = sigmoid(X*Theta1');

    %Add the bias to the result from the Hidden Layer
    a1 = [ones(size(a1,1), 1) a1];

    %Perform the calculations for the output
    a2 = sigmoid(a1*Theta2');
    
    %Find the prediction for the neural network
    [~, p] = max(a2, [], 2);

% =========================================================================


end
