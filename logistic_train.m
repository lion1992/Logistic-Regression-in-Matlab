function [weights] = logistic_train(data,labels,epsilon,maxiterations,SGflag, M, lambda, plotFlag)
% 
% 
% code to train a logistic regression classifier
%
% INPUTS:
% data = n x (d+1) matrix with n samples and d features, where 
% column d+1 is all ones (corresponding to the intercept term) 
%
% labels = n x 1 vector of class labels (taking values 0 or 1) 
%
% epsilon = optional argument specifying the convergence 
% criterion. Let pi be the probability predicted by the model 
% for data point i and deltai be the absolute change in this 
% probability since the last iteration. Let delta be the average 
% of the deltai’s, over all the training data points. When delta 
% becomes lower than epsilon, then halt. 
% (if unspecified, use a default value of 10ˆ-5)
%
% maxiterations = optional argument that specifies the 
% maximum number of iterations to execute (useful when 
% debugging in case your code is not converging correctly!) 
% (if unspecified can be set to 1000) 
%
% SGflag = 1 => use Stochastic gradient descent. The defult would be 0.
%
% M = batch size of stochastic gradient descent.
%
% lambda= step size in stochastic gradient descent.
%
% plotFlag = 1 => plot of accuracies and log-likelihood cost function of 
% each iteration; default value would be 1
% 
% OUTPUT: 
% weights = (d+1) x 1 vector of weights where the weights 
% correspond to the columns of "data"

X=data;
y=labels;

% Set initial weights, a vector with all zeros.
b=zeros(length(X(1,:)),1);  

% Set default values for epsilon and maxiterations.

if(nargin<3 || isempty(epsilon))
    epsilon=10^(-5);
end
if(nargin<4 || isempty(maxiterations))
    maxiterations=1000;
end
if(nargin<5 || isempty(SGflag))
    SGflag=0;
end
if(nargin<8 || isempty(plotFlag))
    plotFlag=1;
end

thetaold=b; % initiate weights.
iter=0; 
done=false;
err=[]; 
loss=[];
n=length(y);
t=[];

tic; % to calculate the time.
while(done==false)
    Pold=1./(1+exp(-X*thetaold));
    % Compute the P(y=1|X,weights) with weights from the previous iteration.
   
    % Newton's Method
    if(SGflag==0)
       
       % Compute the gradient of the log-loss objective function.
       grad=X'*(Pold-y);
       % Compute the diagonal matrix needed for the Hessian matrix. 
       W=diag(exp(X*thetaold)./((1+exp(X*thetaold)).^2));
       % Update the Hessian matrix.
       hessian=X'*W*X;
       % Update the weights. 
       thetanew=thetaold-inv(hessian+10^(-6)*eye(length(X(1,:))))*grad;
    
    % Stochastic Gradient Descent.
    else
       % Generate a random array of size M from 1 to n;
       index=randperm(n,M);
       % Get subset of x;
       x=X(index,:);
       % Get subset of y;
       yt=y(index);
       % Get subset of P(y=1|X,weights) from previously trained weights;
       pold=Pold(index);
       % Compute gradient;
       grad=x.'*(pold-yt);
       % Update weight.
       thetanew=thetaold-lambda*grad;

    end
    % Update the P(y=1|X,weights) with weights from current iteration.
    Pnew=1./(1+exp(-X*thetanew));
    
    % Compute delta's, the absolute change of P
    deltas=abs(Pnew-Pold);
    delta=mean(deltas);
    iter=iter+1;
   
    % Compute the accuracy.
    chat=Pnew>0.5;
    acc=100*sum(y==chat)/length(y);
    err=[err,acc];
    % Compute logloss function.
    logloss=sum(log(1.+exp(X*thetanew)))-y'*X*thetanew;
    loss=[loss, logloss];
    
    if(delta<epsilon || iter>maxiterations)
        done=true;
        weights=thetanew;
    else
        thetaold=thetanew;
    end
    t=[t, toc]; % Record time used for each iteration.
end  
if(plotFlag==1)
    % Plot Accuracy vs. iterations.
    figure;
    plot((1:iter),err,'b-');
    title('Accuracy');
    xlabel('Iteration');
    ylabel('Accuracy (%)');
    
    % Plot Log-loss function vs. iteration.
    figure;
    plot((1:iter),loss, 'r-');
    title('Log-likelihood Cost Function');
    xlabel('Iteration');
    ylabel('Log-likelihood Cost Function ');
    
end
    % Plot Accuracy vs. Time Elapsed.
    figure;
    plot(t,err,'b-');
    title('Performance vs. Time Elapsed');
    xlabel('Time Elapsed (seconds)');
    ylabel('Accuracy');

end
