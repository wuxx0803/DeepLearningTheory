% Helper code with solution for CS229t for HW1 Problem 1
% Author: Sida Wang

function [] = train()
    load MNIST_train2k
    load MNIST_test2k
    
    sigma2 = 50.0;
    kernel = @(x, y) exp(-norm(x - y)^2 / (2 * sigma2));
    
    % kernel = @(x, y) x' * y;
    
    disp('Computing kernel matrix ... ');
    K = zeros(size(X, 2));
    for i = 1:size(X, 2)
        if mod(i, 250) == 0
            disp(['i = ' num2str(i) ' / ' num2str(size(X, 2))]);
        end
        for j = 1:size(X, 2)
            K(i, j) = kernel(X(:, i), X(:, j));
        end
    end
        
    epsilon = 0.01;
    L = chol(K + epsilon * eye(size(K, 1)), 'lower');
    
    Xorig = X;
    X = L';
    
    ytest = ytest - min(ytest) + 1;
    y = y - min(y) + 1;
    numClasses = 10;
    d = size(X,1);
    T = size(X,2);
    W = zeros(numClasses, d);

    %% Train online

    % eta = 0.001; % 0.01, 0.1, 1
    eta = 0.001;

    myLosses = zeros(T, 1);
    myZeroOneLoss = zeros(T, 1);
    
    testZeroOneLoss = [];
    for t=1:T
        if mod(t, 250) == 0
            disp(['t = ' num2str(t) ' / ' num2str(T)]);
        end

        [ftwt, gradt, pred] = ft(W, X(:,t), y(t));
        % [ftu, ~, predu] = ft(U, X(:,t), y(t));
        W = W - eta*gradt;
                
        myLosses(t) = ftwt;
        myZeroOneLoss(t) = double(pred ~= y(t));
        myLossesUbound(t) = norm(gradt, 'fro')^2;
    end
    myLosses = cumsum(myLosses) ./ (1:T)';
    myZeroOneLoss = cumsum(myZeroOneLoss) ./ (1:T)';

%     % plot the quantities required
%     figure;
%     plot(1:T, myLosses, 'r', 'LineWidth', 2);
%     title(['Average hinge losses, eta = ' num2str(eta)], 'FontSize', 12);
%     h_legend = legend('Learner');
%     set(h_legend, 'FontSize', 12);
%     
%     % Correct the axis
%     a = axis;
%     a(4) = 2 * max(myLosses);
%     axis(a);
% 
%     figure;
%     plot(1:T, myZeroOneLoss, 'r', 1:1000:T, testZeroOneLoss, 'ok', 'LineWidth', 2);
%     title(['Average zero-one loss, eta = ' num2str(eta)], 'FontSize', 12);
%     h_legend = legend('Learner', 'Loss on test set');
%     set(h_legend, 'FontSize', 12);

    %% Get train and test accuracies 
    acc = getaccuracy(W, X, y);
    acctest = getaccuracy_new(W, Xorig, L, kernel, Xtest, ytest);
    
    disp(['test accuracy: ' num2str(acctest)]);
    disp(['training accuracy: ' num2str(acc)]);

    %% Use batch optimization to obtain the best expert
    generatebestexpert = 0;
    if generatebestexpert

        objfun = @(Wflat) ft2(reshape(Wflat, numClasses, d), X, y);
        options.Method = 'lbfgs';
        options.DerivativeCheck = 'off';

        U = minFunc(objfun, 1e-10*randn(numClasses*d,1),options);
        U = reshape(U, numClasses, d);
        save('best_expert.mat', 'U')
        outf = fopen('best_expert.txt','w');
        for i=1:length(U(:))
            fprintf(outf, '%d:%f ', i, U(i));
        end
        fclose(outf);

        acc = getaccuracy(U, X, y);
        acctest = getaccuracy(U, Xtest, ytest);
        
        disp(['training accuracy: ' num2str(acc)]);
        disp(['test accuracy: ' num2str(acctest)]);
    end

end

% Takes:
% W is a |Y| by size(X,1).
% X \in R^{dim by N}
% y \in R^N

% Returns:
% f a scalar, the objective function value
% dW, the gradient
% pred, current predictions

function [f, dW, pred] = ft(W, X, y)
    fs = W * X;
    [~, pred] = max(fs);
    fs = 1 + fs - (W(y, :) * X);
    fs(y) = fs(y) - 1;
    f = max(fs);
    
    dW = zeros(size(W));
    if pred ~= y
        dW(pred, :) = X;
        dW(y, :) = -X;
    end
end

% Predict on a new training example
function pred = ft_new(WLi, Xorig, kernel, x)
    c = zeros(size(Xorig, 2), 1);
    for i = 1:length(c)
        c(i) = kernel(Xorig(:, i), x);
    end

    [~, pred] = max(WLi * c);
end

function acc = getaccuracy_new(W, Xorig, L, kernel, Xtest, ytest)
    T = length(ytest);
    preds = zeros(size(ytest));
    WLi = W / L;
    for t = 1:T
        if mod(t, 250) == 0
            disp(['in getaccuracy_new; t = ' num2str(t) ' / ' num2str(T)]);
        end
        pred = ft_new(WLi, Xorig, kernel, Xtest(:, t));
        preds(t) = pred;
    end
    acc = sum(preds == ytest) / length(ytest);
end

function [acc, avgLoss] = getaccuracy(W, X, y)
    T = length(y);
    preds = zeros(size(y));
    lossSum = 0;
%     if your function support matrix X input
%     [~, ~, preds] = ft(W, X, y);
    for t=1:T
        [loss, ~, pred] = ft(W, X(:,t), y(t));
        lossSum = lossSum + loss;
        preds(t) = pred;
    end
    acc = sum(preds == y) / length(y);
    avgLoss = lossSum / T;
end

function [f, dW] = ft2(W, X, y)
    lamb = 0;
    [f, dW] = ft(W, X, y);
    f = f + lamb*0.5*sum(sum(W.*W));
    dW = reshape(dW + lamb*W, [], 1);
end

