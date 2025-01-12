% Load or define dataset
disp('Loading dataset...');
numSamples = 200; % Number of samples
numFeatures = 10; % Number of features

% Generate random feature data and labels
features = rand(numSamples, numFeatures); % Features: [200 x 10]
labels = randi([0, 1], numSamples, 1);    % Labels: Binary classification (0 or 1)

% Normalize the features (Min-Max Scaling)
disp('Normalizing features...');
normalizedFeatures = normalize(features, 'range');
disp('Data normalization completed.');

% Ensure labels are column vectors for binary classification
labels = labels(:); 

% Split data into training and testing sets
disp('Splitting data into training and testing sets...');
trainRatio = 0.8; % Ratio for training data
nSamples = size(normalizedFeatures, 1);
nTrain = round(trainRatio * nSamples);

% Shuffle and split data
randIdx = randperm(nSamples);
trainFeatures = normalizedFeatures(randIdx(1:nTrain), :);
testFeatures = normalizedFeatures(randIdx(nTrain+1:end), :);
trainLabels = labels(randIdx(1:nTrain), :);
testLabels = labels(randIdx(nTrain+1:end), :);

% Transpose for MATLAB neural network compatibility
trainFeatures = trainFeatures';
testFeatures = testFeatures';
trainLabels = trainLabels';
testLabels = testLabels';

% Define the neural network structure with more complexity
disp('Configuring neural network...');
hiddenLayerSizes = [100, 80, 50]; % Increase number of neurons in each layer
net = feedforwardnet(hiddenLayerSizes, 'trainbr'); % Bayesian Regularization for better generalization

% Configure training parameters
net.trainParam.epochs = 1000; % Increased epochs for better training
net.trainParam.goal = 1e-6; % Lower error goal for better performance
net.trainParam.lr = 0.001; % Learning rate decreased for more fine-grained training
net.trainParam.max_fail = 10; % Maximum validation failures to prevent overfitting
net.divideParam.trainRatio = 0.8; % Training data ratio
net.divideParam.valRatio = 0.1; % Validation data ratio
net.divideParam.testRatio = 0.1; % Testing data ratio

% Enable visualization tools
net.trainParam.showWindow = true; % Open training GUI
net.trainParam.showCommandLine = true; % Show detailed command-line output

% Train the network
disp('Training the neural network...');
[net, tr] = train(net, trainFeatures, trainLabels);

% Training completed. Displaying performance diagrams
disp('Training completed. Displaying performance diagrams:');

% Performance Plot
figure;
plotperform(tr); % Performance during training, validation, and testing

% Training State Plot
figure;
plottrainstate(tr); % Shows gradient and validation checks

% Error Histogram
figure;
predictedLabels = net(testFeatures);
errors = testLabels - predictedLabels;
ploterrhist(errors); % Error distribution

% Calculate accuracy
disp('Evaluating accuracy...');
predictedLabels = round(predictedLabels); % For binary classification, round predictions to 0 or 1
accuracy = sum(predictedLabels(:) == testLabels(:)) / numel(testLabels) * 100;
disp(['Accuracy: ', num2str(accuracy), '%']);

% Check if desired accuracy is achieved
if accuracy >= 95
    disp('Desired accuracy of 95% achieved!');
else
    disp('Desired accuracy not achieved. Consider further tuning.');

    
end

% Confusion Matrix for Classification Tasks
figure;
plotconfusion(testLabels, predictedLabels); % Confusion matrix for binary classification
title('Confusion Matrix');

% Plot Error Distribution
errors = testLabels - predictedLabels;
figure;
ploterrhist(errors);
title('Error Distribution');
