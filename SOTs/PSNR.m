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

% Define a wider range of lambda values to increase sparsity
lambda_values = linspace(0.01, 0.5, 30); 

% Initialize arrays to store rates and PSNRs
rates = zeros(length(lambda_values), 1);
PSNRs = zeros(length(lambda_values), 1);

% Keep quantization step size constant
quantization_step = 0.1; 

% Use fixed-length encoding for labels
label_bits_per_block = ceil(log2(K));
label_total_bits = numBlocks * label_bits_per_block;

% Loop over lambda values
for idx = 1:length(lambda_values)
    lambda = lambda_values(idx);

    % Reconstruction and Collect Data for Encoding
    reconstructed_blocks = zeros(size(blocks));
    all_nonzero_coeffs = [];
    all_run_lengths = [];
    all_labels = labels;

    for b = 1:numBlocks
        block = reshape(blocks(:, b), [blockSize, blockSize]);
        k = labels(b); % Retrieve the assigned transform label
        coeffs = H{k}' * block; % Transform coefficients

        % Apply thresholding based on current lambda
        threshold = sqrt(lambda);
        thresholded_coeffs = coeffs;
        thresholded_coeffs(abs(coeffs) < threshold) = 0;

        % Quantize the thresholded coefficients
        quantized_coeffs = round(thresholded_coeffs / quantization_step);

        % Flatten the quantized coefficients in a zig-zag order (optional)
        % For simplicity, we'll use linear indexing
        coeff_vector = quantized_coeffs(:);

        % Run-length encoding of quantized coefficients
        [run_symbols, run_lengths] = RunLengthEncodeCoefficients(coeff_vector);

        % Collect symbols and run lengths
        all_nonzero_coeffs = [all_nonzero_coeffs; run_symbols];
        all_run_lengths = [all_run_lengths; run_lengths];

        % Reconstruct the block
        reconstructed_block = H{k} * (quantized_coeffs * quantization_step); % De-quantize
        reconstructed_blocks(:, b) = reconstructed_block(:); % Flatten and store
    end

    % Convert blocks back to an image
    reconstructed_image = col2im(reconstructed_blocks, [blockSize, blockSize], ...
                                  [rows, cols], 'distinct');

    % Huffman Encoding of run_symbols
    [unique_symbols, ~, symbol_indices] = unique(all_nonzero_coeffs);
    symbol_freqs = accumarray(symbol_indices, 1);

    symbol_probabilities = symbol_freqs / sum(symbol_freqs);
    symbol_dict = huffmandict(unique_symbols, symbol_probabilities);

    symbols_encoded = huffmanenco(all_nonzero_coeffs, symbol_dict);

    % Huffman Encoding of run_lengths
    [unique_lengths, ~, length_indices] = unique(all_run_lengths);
    length_freqs = accumarray(length_indices, 1);

    length_probabilities = length_freqs / sum(length_freqs);
    length_dict = huffmandict(unique_lengths, length_probabilities);

    lengths_encoded = huffmanenco(all_run_lengths, length_dict);

    % Estimate size of Huffman dictionaries
    symbol_dict_size_bits = estimateDictionarySize(symbol_dict, unique_symbols);
    length_dict_size_bits = estimateDictionarySize(length_dict, unique_lengths);

    % Total compressed size
    compressed_size_bits = length(symbols_encoded) + length(lengths_encoded) + ...
                           symbol_dict_size_bits + length_dict_size_bits + label_total_bits;

    % Rate in bits per pixel
    rate_bpp = compressed_size_bits / (rows * cols);
    rates(idx) = rate_bpp;

    % Compute RMSE and PSNR
    original_image_rescaled = image * 255;
    reconstructed_image_rescaled = reconstructed_image * 255;

    rmse = sqrt(mean((original_image_rescaled(:) - reconstructed_image_rescaled(:)).^2));
    PSNR = 20 * log10(255 / rmse);
    PSNRs(idx) = PSNR;

    fprintf('Lambda: %.5f, Rate: %.3f bpp, PSNR: %.2f dB\n', lambda, rate_bpp, PSNR);
end

% Plot PSNR vs Rate
figure;
plot(rates, PSNRs, 'o-');
xlabel('Rate (bits per pixel)');
ylabel('PSNR (dB)');
title('PSNR vs Rate (Varying Lambda, Constant Quantization Step, Using RLE)');
grid on;



% Step 6: Calculate RMSE and Compression Ratio
rmse = sqrt(mean((image(:) - reconstructed_image(:)).^2));
original_size_bits = rows * cols * 8; % Original size in bits (assuming 8 bits per pixel)
compression_ratio = original_size_bits / compressed_size_bits;

% Display results
fprintf('Original image size: %.0f bytes\n', original_size_bits / 8);
fprintf('Compressed image size: %.0f bytes\n', compressed_size_bytes);
fprintf('Compression ratio: %.2f\n', compression_ratio);
fprintf('Final RMSE: %.6f\n', rmse);

% Helper function for Run-Length Encoding of coefficients
function [symbols, run_lengths] = RunLengthEncodeCoefficients(coeff_vector)
    symbols = [];
    run_lengths = [];

    idx = 1;
    while idx <= length(coeff_vector)
        if coeff_vector(idx) == 0
            % Count zeros
            zero_run_length = 0;
            while idx <= length(coeff_vector) && coeff_vector(idx) == 0
                zero_run_length = zero_run_length + 1;
                idx = idx + 1;
            end
            % Store zero run
            symbols = [symbols; 0];
            run_lengths = [run_lengths; zero_run_length];
        else
            % Non-zero coefficient
            symbols = [symbols; coeff_vector(idx)];
            run_lengths = [run_lengths; 1];
            idx = idx + 1;
        end
    end
end

% Helper function to estimate Huffman dictionary size
function dict_size_bits = estimateDictionarySize(dict, symbols)
    dict_size_bits = 0;
    symbol_bits = ceil(log2(max(abs(symbols)) + 1));
    for i = 1:length(dict)
        code_bits = length(dict{i, 2});
        dict_size_bits = dict_size_bits + symbol_bits + code_bits;
    end
end
