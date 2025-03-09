% Step 1: Load the image and preprocess
image = imread('kodak24.png');
if size(image, 3) == 3
    image = rgb2gray(image); % Convert to grayscale if needed
end
image = double(image);
[rows, cols] = size(image);

% Normalize image to range [0, 1]
image = image / 255;

% Step 2: Partition the image into 8x8 blocks
blockSize = 8;
blocks = im2col(image, [blockSize, blockSize], 'distinct'); % Extract 8x8 blocks
[blockRows, numBlocks] = size(blocks);

% Parameters initialization
K = 8; % Number of transform classes (clusters)
lambda_max = 0.01; % Maximum sparsity parameter (annealing starts here)
lambda_min = 0.001; % Minimum sparsity parameter
delta_lambda = 0.001; % Step size for annealing
epsilon = 1e-5; % Convergence tolerance

% Initialize K transforms using KLT-based approach
H = cell(K, 1);

% Step 2.1: Group blocks by gradient-based heuristic
gradient_labels = zeros(1, numBlocks); % Placeholder for gradient labels
for b = 1:numBlocks
    block = reshape(blocks(:, b), [blockSize, blockSize]); % Reshape to 8x8 matrix
    [Gx, Gy] = gradient(block); % Compute gradients
    theta = atan2d(mean(Gy(:)), mean(Gx(:))); % Mean gradient direction
    gradient_labels(b) = mod(round(theta / (180 / K)), K) + 1; % Assign to K classes
end

% Step 2.2: Compute KLT for each class
for k = 1:K
    % Extract blocks assigned to class k
    classBlocks = blocks(:, gradient_labels == k); 
    
    if isempty(classBlocks)
        % If no blocks are assigned, fall back to identity matrix
        H{k} = eye(blockSize);
    else
        % Reshape blocks back to 8x8 for covariance computation
        numClassBlocks = size(classBlocks, 2);
        reshapedBlocks = zeros(blockSize, blockSize, numClassBlocks);
        for i = 1:numClassBlocks
            reshapedBlocks(:, :, i) = reshape(classBlocks(:, i), [blockSize, blockSize]);
        end
        
        % Compute covariance for each pixel location across all blocks
        C = zeros(blockSize, blockSize);
        for i = 1:blockSize
            for j = 1:blockSize
                pixelValues = reshape(reshapedBlocks(i, j, :), [1, numClassBlocks]);
                C(i, j) = cov(pixelValues); % Covariance for the pixel position (i, j)
            end
        end
        C = (C + C') / 2; % Ensure symmetry of the covariance matrix
        [V, D] = eig(C);  % Perform eigen-decomposition
        V = real(V);
        [~, idx] = sort(diag(D), 'descend'); % Sort eigenvalues and eigenvectors by descending order!
        V = V(:, idx); 
        
        % Use eigenvectors as the transform matrix
        H{k} = V;
    end
end

% Step 3: Perform iterative optimization
lambda = lambda_max;
converged = false;

while ~converged
    fprintf('Optimization for lambda = %.4f\n', lambda);
    
    % Step 3.1: Reclassify blocks into K sub-classes
    labels = zeros(1, numBlocks); 
    for b = 1:numBlocks
        block = reshape(blocks(:, b), [blockSize, blockSize]); % Reshape to 8x8 matrix for mult
        costs = zeros(1, K); % Initialize costs for all classes
        for k = 1:K
            coeffs = H{k}' * block; % Transform coefficients
            thresholded_coeffs = coeffs .* (abs(coeffs) >= sqrt(lambda)); % Thresholding!
            % Calculate cost as reconstruction error + sparsity penalty
            costs(k) = norm(block - H{k} * thresholded_coeffs, 'fro')^2 + lambda * nnz(thresholded_coeffs);
        end
        [~, labels(b)] = min(costs); % Assign block to the class with minimum cost
    end

    % Step 3.2: Update transforms for each class
    H_old = H; % Store previous transforms
    for k = 1:K
        classBlocks = blocks(:, labels == k); % Blocks assigned to this transform
        if isempty(classBlocks)
            continue; % Skip if no blocks assigned to this class
        end
        
        % Accumulate covariance matrix for SVD update
        Y = zeros(blockSize, blockSize);
        for b = 1:size(classBlocks, 2)
            block = reshape(classBlocks(:, b), [blockSize, blockSize]); % Reshape to 8x8 matrix
            coeffs = H{k}' * block; % Transform coefficients
            thresholded_coeffs = coeffs .* (abs(coeffs) >= sqrt(lambda)); % Thresholding
            % Outer product for covariance matrix
            Y = Y + block * thresholded_coeffs';
        end
        % Update transform using SVD
        [U, ~, V] = svd(Y, 'econ'); % Singular Value Decomposition
        H{k} = U * V'; % Update with orthonormal transform
    end

    % Check convergence
    transform_diff = 0;
    for k = 1:K
        transform_diff = transform_diff + norm(H{k} - H_old{k}, 'fro'); % Frobenius norm difference
    end
    if transform_diff < epsilon
        converged = true;
    end

    % Annealing step
    lambda = lambda - delta_lambda;
    if lambda < lambda_min
        break;
    end
end

% Your existing steps 1 to 3 remain the same
% ...

% After optimization, proceed to generate PSNR vs Rate curves

% Define quantization step sizes
quantization_steps = linspace(0.01, 1.0, 50); % Adjust as needed

% Initialize arrays to store rates and PSNRs
rates = zeros(length(quantization_steps), 1);
PSNRs = zeros(length(quantization_steps), 1);

% Loop over quantization steps
for q = 1:length(quantization_steps)
    quantization_step = quantization_steps(q);

    % Reconstruction and Collect Data for Encoding
    reconstructed_blocks = zeros(size(blocks));
    all_quantized_coeffs = [];
    all_labels = labels;

    for b = 1:numBlocks
        block = reshape(blocks(:, b), [blockSize, blockSize]); 
        k = labels(b); % Retrieve the assigned transform label
        coeffs = H{k}' * block; % Transform coefficients
        thresholded_coeffs = coeffs .* (abs(coeffs) >= sqrt(lambda_min)); 
        
        % Quantize the thresholded coefficients
        quantized_coeffs = round(thresholded_coeffs / quantization_step);
        
        % Collect all quantized coefficients (including zeros)
        all_quantized_coeffs = [all_quantized_coeffs; quantized_coeffs(:)];
        
        % Reconstruct the block
        reconstructed_block = H{k} * (quantized_coeffs * quantization_step); % De-quantize
        reconstructed_blocks(:, b) = reconstructed_block(:); % Flatten and store
    end

    % Convert blocks back to an image
    reconstructed_image = col2im(reconstructed_blocks, [blockSize, blockSize], ...
                                  [rows, cols], 'distinct');

    % Huffman Encoding
    [coeff_symbols, ~, coeff_indices] = unique(all_quantized_coeffs);
    coeff_freqs = accumarray(coeff_indices, 1);

    coeff_probabilities = coeff_freqs / sum(coeff_freqs);
    coeff_dict = huffmandict(coeff_symbols, coeff_probabilities);

    coeff_encoded = huffmanenco(all_quantized_coeffs, coeff_dict);

    [label_symbols, ~, label_indices] = unique(all_labels);
    label_freqs = accumarray(label_indices, 1);

    label_probabilities = label_freqs / sum(label_freqs);
    label_dict = huffmandict(label_symbols, label_probabilities);

    label_encoded = huffmanenco(all_labels', label_dict);

    % Estimate size of Huffman dictionaries
    coeff_dict_size_bits = 0;
    symbol_bits_coeff = ceil(log2(max(abs(coeff_symbols)) + 1));
    for i = 1:length(coeff_dict)
        code_bits = length(coeff_dict{i, 2});
        coeff_dict_size_bits = coeff_dict_size_bits + symbol_bits_coeff + code_bits;
    end

    label_dict_size_bits = 0;
    symbol_bits_label = ceil(log2(max(abs(label_symbols)) + 1));
    for i = 1:length(label_dict)
        code_bits = length(label_dict{i, 2});
        label_dict_size_bits = label_dict_size_bits + symbol_bits_label + code_bits;
    end

    % Total compressed size
    compressed_size_bits = length(coeff_encoded) + length(label_encoded) + coeff_dict_size_bits + label_dict_size_bits;

    % Rate in bits per pixel
    rate_bpp = compressed_size_bits / (rows * cols);
    rates(q) = rate_bpp;

    % Compute RMSE and PSNR
    original_image_rescaled = image * 255;
    reconstructed_image_rescaled = reconstructed_image * 255;

    rmse = sqrt(mean((original_image_rescaled(:) - reconstructed_image_rescaled(:)).^2));
    PSNR = 20 * log10(255 / rmse);
    PSNRs(q) = PSNR;

    fprintf('Quantization step: %.3f, Rate: %.3f bpp, PSNR: %.2f dB\n', quantization_step, rate_bpp, PSNR);
end

% Plot PSNR vs Rate
figure;
plot(rates, PSNRs, 'o-');
xlabel('Rate (bits per pixel)');
ylabel('PSNR (dB)');
title('PSNR vs Rate');
grid on;


% Step 4: Reconstruction and Collect Data for Encoding
reconstructed_blocks = zeros(size(blocks));

% Initialize arrays to hold all quantized coefficients and labels
all_quantized_coeffs = [];
all_labels = labels;

for b = 1:numBlocks
    block = reshape(blocks(:, b), [blockSize, blockSize]); 
    k = labels(b); % Retrieve the assigned transform label
    coeffs = H{k}' * block; % Transform coefficients
    thresholded_coeffs = coeffs .* (abs(coeffs) >= sqrt(lambda_min)); 
    
    % Quantize the thresholded coefficients
    quantization_step = 0.01; 
    quantized_coeffs = round(thresholded_coeffs / quantization_step);
    
    % Collect non-zero quantized coefficients
    non_zero_coeffs = quantized_coeffs(abs(quantized_coeffs) > 0);
    all_quantized_coeffs = [all_quantized_coeffs; non_zero_coeffs(:)];
    
    % Reconstruct the block
    reconstructed_block = H{k} * (quantized_coeffs * quantization_step); % De-quantize
    reconstructed_blocks(:, b) = reconstructed_block(:); % Flatten and store
end

% Convert blocks back to an image!
reconstructed_image = col2im(reconstructed_blocks, [blockSize, blockSize], ...
                              [rows, cols], 'distinct');

% Step 5: Huffman Encoding (Can replace with ur code!)

% Compute symbol frequencies for coefficients
[coeff_symbols, ~, coeff_indices] = unique(all_quantized_coeffs);
coeff_freqs = accumarray(coeff_indices, 1);

% Generate Huffman dictionary for coefficients
coeff_probabilities = coeff_freqs / sum(coeff_freqs);
coeff_dict = huffmandict(coeff_symbols, coeff_probabilities);

% Encode coefficients using Huffman encoding
coeff_encoded = huffmanenco(all_quantized_coeffs, coeff_dict);

% Compute symbol frequencies for labels
[label_symbols, ~, label_indices] = unique(all_labels);
label_freqs = accumarray(label_indices, 1);

% Generate Huffman dictionary for labels
label_probabilities = label_freqs / sum(label_freqs);
label_dict = huffmandict(label_symbols, label_probabilities);

% Encode labels using Huffman encoding
label_encoded = huffmanenco(all_labels', label_dict);

% Estimate size of Huffman dictionaries
coeff_dict_size_bits = 0;
for i = 1:length(coeff_dict)
    symbol_bits = 32; % Assuming 32 bits to store the symbol 
    code_bits = length(coeff_dict{i, 2});
    coeff_dict_size_bits = coeff_dict_size_bits + symbol_bits + code_bits;
end

label_dict_size_bits = 0;
for i = 1:length(label_dict)
    symbol_bits = 32; % Assuming 32 bits to store the symbol
    code_bits = length(label_dict{i, 2});
    label_dict_size_bits = label_dict_size_bits + symbol_bits + code_bits;
end

% Total compressed size
compressed_size_bits = length(coeff_encoded) + length(label_encoded) + coeff_dict_size_bits + label_dict_size_bits;
compressed_size_bytes = compressed_size_bits / 8;

% Step 6: Calculate RMSE and Compression Ratio
rmse = sqrt(mean((image(:) - reconstructed_image(:)).^2));
original_size_bits = rows * cols * 8; % Original size in bits (assuming 8 bits per pixel)
compression_ratio = original_size_bits / compressed_size_bits;

% Display results
fprintf('Original image size: %.0f bytes\n', original_size_bits / 8);
fprintf('Compressed image size: %.0f bytes\n', compressed_size_bytes);
fprintf('Compression ratio: %.2f\n', compression_ratio);
fprintf('Final RMSE: %.6f\n', rmse);

% Visualize original and reconstructed images
figure;
subplot(1, 2, 1);
imshow(image, []);
title('Original Image');

subplot(1, 2, 2);
imshow(reconstructed_image, []);
title('Reconstructed Compressed Image');
