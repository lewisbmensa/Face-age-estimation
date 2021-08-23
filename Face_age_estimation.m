%% information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;


% path 
database_path = 'C:\Users\lewis\Documents\MATLAB\ecs797\ECS797Lab3\data_age.mat\';
result_path = 'C:\Users\lewis\Documents\MATLAB\ecs797\ECS797Lab3\Results';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 
error = ytest - yhat_test;

%using a built in function to determine the mean asbolute error
mae = mean(abs(error));

%find the absolute error for the predicitions
abs_error = abs(error);

%find the rows where the absolute error is less than or equal to 5
row = find(abs_error <= 5);

%return the abs values of all the rows with an absolute error of 5 or less
N_e_values = abs_error(row,:);

%find the number of values with an absolute error of 5 or less
N_e = length(N_e_values);

%find the number of values in our dataset
N_count = length(error);

%find the cs with error level 5
cs_number = (N_e/(N_count) * 100);




%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
CS_vector = zeros(1,15);


N_e_new = zeros(1,15);


for error_value = 1:15
    row_n = find(abs_error<=error_value);
    N_e_new(error_value) = length(row_n);
    CS_vector(error_value) = N_e_new(error_value) *(1/N_count)*100;

end

plot(1:15,CS_vector);
xlabel('Error_value');
ylabel('Cumaluative score');






    


    
    


%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.

%training the PLS model using 10 components
[XL,yl,XS,YS,beta,PCTVAR] = plsregress(xtrain,ytrain,10);

%predicting the test set using the PLS model
PLS_pred = [ones(size(xtest,1),1) xtest]*beta;

%compute the error of the PLS model
PLS_error = ytest - PLS_pred ;

%mean absolute error of the PLS model
pls_mae = mean(abs(PLS_error));


%find the absolute error for the predicitions of the PLS regression
abs_PLS_error = abs(PLS_error);


%find the rows where the absolute error is less than or equal to 5
PLS_row = find(abs_PLS_error <= 5);

%return the abs values of all the rows with an absolute error of 5 or less
N_e_PLS_values = abs_PLS_error(PLS_row,:);

%find the number of values with an absolute error of 5 or less
N_e_PLS = length(N_e_PLS_values);

%find the cs with error level 5
PLS_cs_number = (N_e_PLS/(N_count) * 100);

%training the regression tree model
tree = fitrtree(xtrain,ytrain);

%predict the test data using the tree model
pred_tree = predict(tree,xtest);

%find the error between the tree prediction and the ground truth
tree_error = ytest - pred_tree;

%find the mean absolute error 
tree_mae = mean(abs(tree_error));

%find the absolute error between the tree model and the ground truth
abs_tree_error = abs(tree_error);

%find the index where the error level is 5 or less
tree_row = find(abs_tree_error <= 5);

%find the tree values where the error level is 5 or less
N_e_tree_values = abs_tree_error(tree_row,:);

%find the number of values with an absolute error of 5 or less
N_e_tree = length(N_e_tree_values);

%CS for the tree model
tree_cs_number = (N_e_tree/(N_count) * 100);



%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox

%train the svm model
model = fitrsvm(xtrain,ytrain,'Standardize',true);

%predict the test data using the model
svm = predict(model,xtest);

%find the error of the model
svm_error = ytest - svm;

%find the mean absolute error for the model
svm_mae = mean(abs(svm_error));

%find the absolute error for the model
abs_svm_error = abs(svm_error);

%find the index where the error level is 5 or less
svm_row = find(abs_svm_error <= 5);

%find the svm values where the error level is 5 or less
N_e_svm_values = abs_svm_error(svm_row,:);

%%find the number of values with an absolute error of 5 or less
N_e_svm = length(N_e_svm_values);

%CS for the svm model
svm_cs_number = (N_e_svm/(N_count) * 100);





