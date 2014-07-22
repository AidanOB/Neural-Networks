function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

    
    m = length(y); % number of training examples

    %Calling the non regularised cost Function
    g = sigmoid(X*theta);
    J = (1/m)*sum(-y.*log(g) - (1-y).*log(1-g));
    error = g - y;
    [~, n] = size(X);
    errVec = error(:,ones(1,n));
    grad = (1/m) .* sum(errVec .* X);

    %Adding the regularisation to the cost
    J = J + (lambda/(2*m))*sum(theta(2:end).^2);

    %Regularising the gradient. For j = 0
    grad(2:end) = grad(2:end) + lambda/m * theta(2:end)';

% =============================================================

    grad = grad(:);

end
