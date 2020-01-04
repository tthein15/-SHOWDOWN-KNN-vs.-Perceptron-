% Importing and Cleaning
%currencytrain1 = readtable('currencytrain.csv'); % import currencytrain.csv
%currencytrain1.note = categorical(currencytrain1.note); % converting note row to categorical
%cleaned = currencytrain1(:,[2:end 1]); % moving real/fake label to last column
%[trainDev, testData] = splitTrainTest(cleaned , 0.8); % splits Data into Testing and Training/Development
%view(trainedModel1, 'mode', 'graph')

% Decision Tree: Feature Selection 
%currencytrain1 = readtable('currencytrain.csv')
%test = readtable('currencytrain.csv')
%clc
% Reproducibility
rng(1); 

Dtree_Model = fitctree(trainDev,'note','PredictorSelection','curvature',...
    'Surrogate','off');
imp = predictorImportance(Dtree_Model);

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Dtree_Model.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

Dtree_Model2 = fitctree(trainDev,'note','CrossVal','on');
view(Dtree_Model2.Trained{1}, 'Mode', 'graph'); % variance between just diag_length and diag_length + bottom_margin

Dtree_Model3 = fitctree(trainDev,'note','OptimizeHyperparameters','MaxNumSplits',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
%view(dtModel3, 'Mode', 'graph');

Dtree_Model4 = fitctree(trainDev,'note', 'MaxNumSplits', 2);
% variance between just diag_length and diag_length + bottom_margin
view(Dtree_Model4, 'Mode', 'graph');

dtModel5 = fitctree(zscore(trainDev{:,1:end-1}),trainDev.note, 'MaxNumSplits', 2);
% variance between just diag_length and diag_length + bottom_margin
view(dtModel5, 'Mode', 'graph'); 

dtModelBootStrapped = TreeBagger(10, trainDev{:,1:end-1},trainDev.note,'MaxNumSplits', 2, 'OOBPredictorImportance','On');
figure
plot(oobError(dtModelBootStrapped))
xlabel('Number of Trees')
ylabel('Out-of-Bag Classification Error')

knn_Model1 = fitcknn(trainDev, 'note', 'OptimizeHyperparameters','NumNeighbors',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus')) % 'auto' ? Use {'Distance','NumNeighbors'}.

knn_Model2 = fitcknn(trainDev, 'note', 'NumNeighbors', 37);

knnModel_CrossVal = crossval(knn_Model2);
class_error = kfoldLoss(knnModel_CrossVal)



% feature selection (modify column parameter)
trainDev_FeatSelect = trainDev(:,1:end); 
[train_Perceptron, ~] = splitTrainTest(trainDev_FeatSelect , 1); 
kfolds = 10;
maxIter = 50;
stdev = 1;
percept1 = createPerceptron(train_Perceptron, maxIter, kfolds, stdev);
sum_Error = 0;
for i = 1:length(percept1)
    sum_Error = sum_Error + percept1{i}.class_error;
end

% avg mean absolute error
disp(sum_Error/kfolds) 
disp(percept1{1}.performanceSummary)


% feature selection (modify column parameter)
trainDev_FeatSelect = trainDev(:,5:end); 
[train_Perceptron, devPerceptron] = splitTrainTest(trainDev , 1); 
kfolds = 10;
maxIter = 50;
stdev = 1;
percept2 = createPerceptron(train_Perceptron, maxIter, kfolds, stdev);
sum_Error = 0;
for i = 1:length(percept2)
    sum_Error = sum_Error + percept2{i}.classError;
end

% avg mean absolute error
disp(sum_Error/kfolds)

%[predictions, classError] = predict(percept{1}, devPerceptron);



Dtree_Predictions  = predict(Dtree_Model4, testData{:,1:end-1});
Dtree_numCorrect = 0;
for i_row = 1:length(Dtree_Predictions)
    if (testData{i_row,end} == 'real') == (Dtree_Predictions(i_row,1) == 'real')
        Dtree_numCorrect = Dtree_numCorrect + 1;
    elseif (testData{i_row,end} == 'fake') == (Dtree_Predictions(i_row,1) == 'fake')
        Dtree_numCorrect = Dtree_numCorrect + 1;
    else
    end
end
disp(Dtree_numCorrect)

knnPredictions  = predict(knn_Model2, testData{:,1:end-1});
knn_numCorrect = 0;
for i_row = 1:length(knnPredictions)
    if (testData{i_row,end} == {'real'}) == (knnPredictions(i_row,1) == 'real')
        knn_numCorrect = knn_numCorrect + 1;
    elseif (testData{i_row,end} == {'fake'}) == (knnPredictions(i_row,1) == 'fake')
        knn_numCorrect = knn_numCorrect + 1;
    else
    end
end
disp(knn_numCorrect)

perceptron_Predictions = predictPerceptron(percept1{1}, testData(:,1:end));
percept_numCorrect = 0;
for i_row = 1:height(perceptron_Predictions)
    if (testData{i_row,end} == 'real') == (perceptron_Predictions{i_row,1} == 'real')
        percept_numCorrect = percept_numCorrect + 1;
    elseif (testData{i_row,end} == 'fake') == (perceptron_Predictions{i_row,1} == 'fake')
        percept_numCorrect = percept_numCorrect + 1;
    else
    end
end

disp(percept_numCorrect)
str = sprintf('DT Error: %4.f. KNN Error: %4.f, Perceptron Error: %4.f.', 1-Dtree_numCorrect/height(testData), 1-knn_numCorrect/height(testData), 1-(percept_numCorrect/height(testData)));
disp(str)
view(MdlDefault, 'Mode', 'graph');

%{
for maxIter = 1:50
    str = perceptronRun(trainDev, maxIter);
    disp(str)
end
%}
%}