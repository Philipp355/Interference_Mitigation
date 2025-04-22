
% calculateErrors.m
function [range_error, velocity_error] = calculateErrors(range_estimates, ...
    velocity_estimates, targets)
    
    num_targets = length(targets);
    num_estimates = length(range_estimates);
    
    % Initialize error arrays
    range_error = zeros(1, num_targets);
    velocity_error = zeros(1, num_targets);

%     % Debug print
%     fprintf('Number of targets: %d\n', num_targets);
%     fprintf('Number of estimates: %d\n', num_estimates);
    
    % find the closest estimate
    for i = 1:num_targets
        true_range = targets(i).range;
        true_velocity = targets(i).velocity;

        range_distances = abs(range_estimates - true_range);
        velocity_distances = abs(velocity_estimates - true_velocity);

        total_distances = range_distances/max(abs(true_range), 1) + ...
            velocity_distances/max(abs(true_velocity), 1);
        
        if ~isempty(total_distances)
            [min_dist, min_idx] = min(total_distances);
            
            if min_dist < 5  
                range_error(i) = range_estimates(min_idx) - true_range;
                velocity_error(i) = velocity_estimates(min_idx) - true_velocity;
            else
                % if no close estimate found
                range_error(i) = NaN;
                velocity_error(i) = NaN;
            end
        else
            range_error(i) = NaN;
            velocity_error(i) = NaN;
        end
    end
end
