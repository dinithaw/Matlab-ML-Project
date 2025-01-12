% Load or define dataset
% Replace this with your actual dataset loading code
% Example of random data generation for testing
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

% Check and prepare labels
if isnumeric(labels) && numel(unique(labels)) > 2
    % One-hot encode labels for multi-class classification
    labels = ind2vec(labels')'; 
elseif isnumeric(labels)
    % For binary classification, ensure labels are column vectors
    labels = labels(:); 
end

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

% Define the neural network structure
disp('Configuring neural network...');
hiddenLayerSizes = [50, 30, 20]; % Enhanced structure for better learning
net = feedforwardnet(hiddenLayerSizes);

% Configure training parameters
net.trainParam.epochs = 1000; % Increased epochs for better training
net.trainParam.goal = 1e-4; % Lower error goal
net.trainParam.lr = 0.01; % Learning rate
net.trainParam.max_fail = 20; % Maximum validation failures
net.divideParam.trainRatio = 0.8; % Training data ratio
net.divideParam.valRatio = 0.1; % Validation data ratio
net.divideParam.testRatio = 0.1; % Testing data ratio

% Enable visualization tools
net.trainParam.showWindow = true; % Open training GUI
net.trainParam.showCommandLine = true; % Show detailed command-line output

% Train the network
disp('Training the neural network...');
[net, tr] = train(net, trainFeatures, trainLabels);

% Display training diagrams
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
if size(testLabels, 1) > 1
    % Convert one-hot encoded labels to class indices
    [~, predictedLabels] = max(predictedLabels, [], 1);
    [~, trueLabels] = max(testLabels, [], 1);
else
    % For regression or binary classification
    trueLabels = testLabels;
end
errors = trueLabels(:) - predictedLabels(:);
ploterrhist(errors); % Error distribution

% Regression Plot (Fixed)
figure;
try
    if isnumeric(trueLabels) && isnumeric(predictedLabels)
        % Ensure proper shape and limits
        trueLabels = trueLabels(:);
        predictedLabels = predictedLabels(:);

        % Check if trueLabels and predictedLabels are constant
        if all(trueLabels == trueLabels(1)) && all(predictedLabels == predictedLabels(1))
            % Add a small range to avoid identical axis limits
            trueLabels = trueLabels + randn(size(trueLabels)) * 1e-6;
            predictedLabels = predictedLabels + randn(size(predictedLabels)) * 1e-6;
        end

        % Set axis limits explicitly
        minVal = min([trueLabels; predictedLabels]);
        maxVal = max([trueLabels; predictedLabels]);
        if minVal == maxVal
            minVal = minVal - 1; % Adjust for identical values
            maxVal = maxVal + 1;
        end

        % Call plotregression with corrected data
        plotregression(trueLabels, predictedLabels);
        xlim([minVal, maxVal]); % Set axis limits
        ylim([minVal, maxVal]); % Ensure square plot
    else
        warning('plotregression is only applicable for numeric data.');
    end
catch ME
    disp('An error occurred during regression plotting:');
    disp(ME.message);
end

% Calculate accuracy
disp('Evaluating accuracy...');
if size(testLabels, 1) > 1
    % Multi-class classification
    [~, predictedLabels] = max(predictedLabels, [], 1);
    [~, trueLabels] = max(testLabels, [], 1);
end
accuracy = sum(predictedLabels(:) == trueLabels(:)) / numel(trueLabels) * 100;
disp(['Accuracy: ', num2str(accuracy), '%']);

% Check if desired accuracy is achieved
if accuracy >= 95
    disp('Desired accuracy of 95% achieved!');
else
    disp('Desired accuracy not achieved. Consider further tuning.');
end


% Test the trained model
predictions = net(testFeatures');
[~, predictedLabels] = max(predictions, [], 1); % Assuming classification problem

% Calculate accuracy
accuracy = sum(predictedLabels' == testLabels) / length(testLabels) * 100;
disp(['Model Accuracy: ', num2str(accuracy), '%']);

% Plot confusion matrix
figure;
plotconfusion(ind2vec(testLabels'), ind2vec(predictedLabels'));
title('Confusion Matrix');
