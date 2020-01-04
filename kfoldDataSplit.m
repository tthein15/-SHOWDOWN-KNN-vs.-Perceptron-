function [training, development] = kfoldDataSplit(Data , kfolds)
% splitTrainTest: remodified for train/dev info split for kfolds
% initialize length of cell array to kfolds
training(1:kfolds) = {'temp'};
development(1:kfolds) = {'temp'};
% obtaining height of Data
[m,~] = size(Data); 
% randomly permute the indexes
idx_perm = randperm(m); 
% initializing Row Index variable
row_Idx = 1; 
% iterating k times for kfolds
for kfoldCounter = 1:kfolds 
    row_Idx = row_Idx + round((1/kfolds)*m);
    % IndexOutofBound error(Best Case)
    if row_Idx+1>height(Data) 
        development{kfoldCounter} = Data(idx_perm(row_Idx-round((1/kfolds)*m):end),:); 
        training{kfoldCounter} = Data(idx_perm(1:row_Idx-round((1/kfolds)*m)-1),:);
    % training/dev split   
    else 
        development{kfoldCounter} = Data(idx_perm(row_Idx-round((1/kfolds)*m):row_Idx-1),:); 
        training{kfoldCounter} = Data(idx_perm([1:row_Idx-round((1/kfolds)*m)-1, row_Idx:end]),:);
    end
end
%disp(testing)
%disp(training)
end