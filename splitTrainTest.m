function [training, testing] = splitTrainTest(data_Set , P)

[m,~] = size(data_Set);
idx_perm = randperm(m);
training = data_Set(idx_perm(1:round(P*m)),:); 
testing = data_Set(idx_perm(round(P*m)+1:end),:);

% https://www.mathworks.com/matlabcentral/answers/395136-how-to-divide-a-data-set-randomly-into-training-and-testing-data-set

end