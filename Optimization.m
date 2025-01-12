% Step 1: Load and Normalize Data
disp('Loading dataset...');
numSamples = 1000; % Dataset size
numFeatures = 10; % Number of features

% Generate random features and labels (binary classification)
features = rand(numSamples, numFeatures);
labels = randi([0, 1], numSamples, 1);

% Normalize features
disp('Normalizing features...');
normalizedFeatures = normalize(features, 'range');
labels = labels(:);

% Step 2: Balance the Dataset
disp('Balancing dataset...');
numClass0 = sum(labels == 0);
numClass1 = sum(labels == 1);

if numClass0 > numClass1
    minorityFeatures = normalizedFeatures(labels == 1, :);
    minorityLabels = labels(labels == 1);
    oversampleFactor = ceil((numClass0 - numClass1) / numClass1);
    featuresBalanced = [normalizedFeatures; repmat(minorityFeatures, oversampleFactor, 1)];
    labelsBalanced = [labels; repmat(minorityLabels, oversampleFactor, 1)];
elseif numClass1 > numClass0
    minorityFeatures = normalizedFeatures(labels == 0, :);
    minorityLabels = labels(labels == 0);
    oversampleFactor = ceil((numClass1 - numClass0) / numClass0);
    featuresBalanced = [normalizedFeatures; repmat(minorityFeatures, oversampleFactor, 1)];
    labelsBalanced = [labels; repmat(minorityLabels, oversampleFactor, 1)];
else
    featuresBalanced = normalizedFeatures;
    labelsBalanced = labels;
end

% Shuffle the dataset
disp('Shuffling balanced dataset...');
randIdx = randperm(size(featuresBalanced, 1));
featuresBalanced = featuresBalanced(randIdx, :);
labelsBalanced = labelsBalanced(randIdx);

% Transpose for MATLAB compatibility
featuresBalanced = featuresBalanced';
labelsBalanced = labelsBalanced';

% Step 3: Split Data into Training and Testing Sets
disp('Splitting dataset...');
trainRatio = 0.8;
numSamples = size(featuresBalanced, 2);
nTrain = round(trainRatio * numSamples);
trainFeatures = featuresBalanced(:, 1:nTrain);
testFeatures = featuresBalanced(:, nTrain+1:end);
trainLabels = labelsBalanced(1:nTrain);
testLabels = labelsBalanced(nTrain+1:end);

% Step 4: Configure Neural Network
disp('Configuring neural network...');
hiddenLayerSizes = [100, 80, 60]; % Three-layer architecture
net = feedforwardnet(hiddenLayerSizes, 'trainlm'); % Levenberg-Marquardt

% Training parameters
net.trainParam.epochs = 1500; % More epochs for convergence
net.trainParam.goal = 1e-5; % Lower error goal
net.trainParam.lr = 0.001; % Learning rate for fine-grained optimization
net.trainParam.max_fail = 10; % Maximum validation failures
net.divideParam.trainRatio = 0.7; % Training ratio
net.divideParam.valRatio = 0.2; % Validation ratio
net.divideParam.testRatio = 0.1; % Testing ratio

% Step 5: Train the Network
disp('Training neural network...');
[net, tr] = train(net, trainFeatures, trainLabels);

% Step 6: Evaluate the Model
disp('Evaluating the model...');
predictions = net(testFeatures);
predictedLabels = round(predictions); % Round for binary classification

% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
disp(['Model Accuracy: ', num2str(accuracy), '%']);

% Optimization Visualization
figure;
hold on;
plot(tr.epoch, tr.perf, '-o', 'DisplayName', 'Training Loss');
plot(tr.epoch, tr.vperf, '-x', 'DisplayName', 'Validation Loss');
xlabel('Epochs');
ylabel('Loss');
title('Optimization Performance');
legend('show');
grid on;
hold off;

% Step 7: Display Results
if accuracy >= 95
    disp('Desired accuracy of 95% achieved!');
else
    disp('Accuracy below 95%. Consider further adjustments.');
end

% Confusion Matrix
figure;
plotconfusion(testLabels, predictedLabels); % Confusion matrix
title('Confusion Matrix');

% Error Histogram
figure;
ploterrhist(testLabels - predictedLabels); % Error histogram
title('Error Histogram');
