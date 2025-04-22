
function [pd, fa] = compareDetections(clean_detections, test_detections, config)
    % Find connected components in both detection maps
    clean_labels = bwlabel(clean_detections);
    test_labels = bwlabel(test_detections);
    
    % Get properties of each detection cluster
    clean_props = regionprops(clean_labels, 'Centroid');
    test_props = regionprops(test_labels, 'Centroid');
    
    num_true = length(clean_props);
    num_detected = length(test_props);

    true_positives = 0;
    false_alarms = 0;
    
    max_distance = 5;  % adjust as needed
    
    % For each true target, find closest detection
    for i = 1:num_true
        true_centroid = clean_props(i).Centroid;
        
        % closest detection
        min_dist = inf;
        matched = false;
        
        for j = 1:num_detected
            test_centroid = test_props(j).Centroid;
            dist = norm(true_centroid - test_centroid);
            
            if dist < min_dist && dist < max_distance
                min_dist = dist;
                matched = true;
            end
        end
        
        if matched
            true_positives = true_positives + 1;
        end
    end
    
    % Calculate false alarms (detections not matching any true target)
    for j = 1:num_detected
        test_centroid = test_props(j).Centroid;
        
        min_dist = inf;
        for i = 1:num_true
            true_centroid = clean_props(i).Centroid;
            dist = norm(test_centroid - true_centroid);
            min_dist = min(min_dist, dist);
        end
        
        if min_dist > max_distance
            false_alarms = false_alarms + 1;
        end
    end
    
    pd = true_positives / num_true;  
    fa = false_alarms / numel(clean_detections);  
    
    fprintf('Detection metrics:\n');
    fprintf('  True targets: %d\n', num_true);
    fprintf('  Detections: %d\n', num_detected);
    fprintf('  True positives: %d\n', true_positives);
    fprintf('  False alarms: %d\n', false_alarms);
    fprintf('  PD: %.2f\n', pd);
    fprintf('  FA: %.2e\n', fa);
end


function [range_idx, doppler_idx] = convertToIndices(range, velocity, range_resolution, velocity_resolution, map_size)
    % Convert range to index
    range_idx = round(range / range_resolution) + 1;
    
    doppler_idx = round(velocity / velocity_resolution) + floor(map_size(2)/2) + 1;
    
    range_idx = max(1, min(range_idx, map_size(1)));
    doppler_idx = max(1, min(doppler_idx, map_size(2)));
end