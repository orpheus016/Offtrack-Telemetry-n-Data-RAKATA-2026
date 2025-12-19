clear; clc;
%% ========================================================================
%  NARX FOR SEM LAP STRATEGY OPTIMIZATION - SINGLE OPTIMAL STRATEGY
%  
%  Input to NARX:  
%    - Test drive parameters (lap time target, energy budget, aggressiveness)
%    - Track features (elevation, curvature from GPS)
%  
%  Output from NARX (ONE OPTIMAL STRATEGY):
%    - Speed upper bound (km/h)
%    - Speed lower bound (km/h)  
%    - Throttle start locations (distance in m)
%    - Gliding start locations (distance in m)
% ========================================================================

%% INITIALIZATION
basePath = 'D:\.kuliah\S1\rakata\2025';
telemetryFile = fullfile(basePath, '26novfiltered.csv');
logbookFile = fullfile(basePath, 'Logbook_TD_26_10.csv');

assert(isfile(telemetryFile), 'Telemetry file not found!');
assert(isfile(logbookFile), 'Logbook file not found!');

%% LOAD TELEMETRY DATA
fprintf('=== LOADING TELEMETRY DATA ===\n');
T = readtable(telemetryFile);

lat_raw = double(T.lat);
lng_raw = double(T.lng);
speed_raw = double(T.kecepatan);
throttle_raw = double(T.throttle);
current_raw = double(T.arus);
millis_raw = double(T.millis);

% Remove invalid GPS (0,0)
valid = ~(lat_raw == 0 & lng_raw == 0);
lat_raw = lat_raw(valid);
lng_raw = lng_raw(valid);
speed_raw = speed_raw(valid);
throttle_raw = throttle_raw(valid);
current_raw = current_raw(valid);
millis_raw = millis_raw(valid);

fprintf('Raw data points: %d\n', length(lat_raw));

%% GPS GLITCH FILTER - INDONESIA ONLY
fprintf('\n=== GPS FILTERING (INDONESIA BOUNDS) ===\n');

% Indonesia bounding box
LAT_MIN = -11.0; LAT_MAX = 6.0;
LNG_MIN = 95.0;  LNG_MAX = 141.0;

MAX_JUMP_DISTANCE = 1000; % meters
MIN_DISTANCE = 0.1;

haversine = @(lat1, lon1, lat2, lon2) ...
    6371000 * 2 * asin(sqrt(sind((lat2-lat1)/2).^2 + ...
    cosd(lat1) .* cosd(lat2) .* sind((lon2-lon1)/2).^2));

N_raw = length(lat_raw);
is_valid = true(N_raw, 1);

% Filter 1: Geographic bounds
out_of_bounds = lat_raw < LAT_MIN | lat_raw > LAT_MAX | ...
                lng_raw < LNG_MIN | lng_raw > LNG_MAX;
is_valid = is_valid & ~out_of_bounds;
fprintf('Points outside Indonesia: %d\n', sum(out_of_bounds));

% Filter 2: Distance jumps
distances = zeros(N_raw-1, 1);
for i = 1:N_raw-1
    distances(i) = haversine(lat_raw(i), lng_raw(i), ...
                            lat_raw(i+1), lng_raw(i+1));
end

for i = 1:N_raw-1
    if distances(i) > MAX_JUMP_DISTANCE
        is_valid(i) = false;
        is_valid(i+1) = false;
        fprintf('Glitch: %.1f km jump at point %d\n', distances(i)/1000, i);
    end
end

% Apply filter
lat = lat_raw(is_valid);
lng = lng_raw(is_valid);
speed = speed_raw(is_valid);
throttle = throttle_raw(is_valid);
current = current_raw(is_valid);
millis = millis_raw(is_valid);

fprintf('Filtered: %d points (%.1f%% retained)\n', ...
    length(lat), length(lat)/N_raw*100);

%% CALCULATE DISTANCE
fprintf('\n=== CALCULATING DISTANCE ===\n');

N = length(lat);
R = 6371000;

dlat = deg2rad(diff(lat));
dlng = deg2rad(diff(lng));
a = sin(dlat/2).^2 + cos(deg2rad(lat(1:end-1))) .* ...
    cos(deg2rad(lat(2:end))) .* sin(dlng/2).^2;
d = 2 * R * atan2(sqrt(a), sqrt(1-a));
d(d < MIN_DISTANCE) = MIN_DISTANCE;
distance = [0; cumsum(d)];

total_distance = distance(end);
fprintf('Total distance: %.2f km\n', total_distance/1000);

%% CALCULATE ROAD SLOPE
roadSlope = [0; diff(speed) ./ diff(distance)];
roadSlope(~isfinite(roadSlope)) = 0;
roadSlope_pct = roadSlope * 100;

%% CALCULATE CURVATURE
curvature = zeros(N, 1);
for i = 2:N-1
    y1 = sind(lng(i)-lng(i-1)) * cosd(lat(i));
    x1 = cosd(lat(i-1)) * sind(lat(i)) - sind(lat(i-1)) * cosd(lat(i)) * cosd(lng(i)-lng(i-1));
    bearing1 = atan2d(y1, x1);
    
    y2 = sind(lng(i+1)-lng(i)) * cosd(lat(i+1));
    x2 = cosd(lat(i)) * sind(lat(i+1)) - sind(lat(i)) * cosd(lat(i+1)) * cosd(lng(i+1)-lng(i));
    bearing2 = atan2d(y2, x2);
    
    bearing_change = abs(bearing2 - bearing1);
    if bearing_change > 180
        bearing_change = 360 - bearing_change;
    end
    curvature(i) = bearing_change;
end
curvature = smoothdata(curvature, 'gaussian', 10);

%% LOAD LOGBOOK
fprintf('\n=== LOADING LOGBOOK ===\n');
LB = readtable(logbookFile);

throttleCols = contains(lower(LB.Properties.VariableNames), 'throtle') & ...
               contains(lower(LB.Properties.VariableNames), '_s');
glidingCols = contains(lower(LB.Properties.VariableNames), 'gliding') & ...
              contains(lower(LB.Properties.VariableNames), '_s');

throttleVars = LB.Properties.VariableNames(throttleCols);
glidingVars = LB.Properties.VariableNames(glidingCols);

eventThrottle = [
    LB.(throttleVars{1})
    LB.(throttleVars{2})
    LB.(throttleVars{3})
    LB.(throttleVars{4})
];

eventGliding = [
    LB.(glidingVars{1})
    LB.(glidingVars{2})
    LB.(glidingVars{3})
    LB.(glidingVars{4})
];

eventThrottle = eventThrottle(~isnan(eventThrottle));
eventGliding = eventGliding(~isnan(eventGliding));

fprintf('Throttle events: %d\n', length(eventThrottle));
fprintf('Gliding events: %d\n', length(eventGliding));

%% MAP LOGBOOK EVENTS TO DISTANCE
fprintf('\n=== MAPPING EVENTS TO DISTANCE ===\n');

time_s = (millis - millis(1)) / 1000;

% Find distance at each throttle/gliding event
throttle_distances = zeros(length(eventThrottle), 1);
gliding_distances = zeros(length(eventGliding), 1);

for i = 1:length(eventThrottle)
    [~, idx] = min(abs(time_s - eventThrottle(i)));
    throttle_distances(i) = distance(idx);
end

for i = 1:length(eventGliding)
    [~, idx] = min(abs(time_s - eventGliding(i)));
    gliding_distances(i) = distance(idx);
end

fprintf('Throttle start locations (m): ');
fprintf('%.1f ', throttle_distances);
fprintf('\n');
fprintf('Gliding start locations (m): ');
fprintf('%.1f ', gliding_distances);
fprintf('\n');

%% CREATE THROTTLE STATE FROM LOGBOOK
throttle_state_actual = zeros(N, 1);

% Mark throttle ON periods
for i = 1:length(throttle_distances)
    t_start_dist = throttle_distances(i);
    
    if i <= length(gliding_distances)
        t_end_dist = gliding_distances(i);
    else
        t_end_dist = distance(end);
    end
    
    idx_on = find(distance >= t_start_dist & distance < t_end_dist);
    throttle_state_actual(idx_on) = 1;
end

fprintf('Throttle ON: %.1f%% of track\n', sum(throttle_state_actual)/N*100);

%% CALCULATE ACTUAL SPEED WINDOWS
fprintf('\n=== CALCULATING ACTUAL SPEED WINDOWS ===\n');

segment_length = 100; % meters
num_segments = ceil(total_distance / segment_length);

speed_upper_actual = zeros(N, 1);
speed_lower_actual = zeros(N, 1);

for seg = 1:num_segments
    seg_start = (seg-1) * segment_length;
    seg_end = seg * segment_length;
    seg_idx = find(distance >= seg_start & distance < seg_end);
    
    if isempty(seg_idx)
        continue;
    end
    
    seg_speeds = speed(seg_idx);
    seg_mean = mean(seg_speeds);
    seg_std = std(seg_speeds);
    
    margin = max(seg_std, 3);
    
    speed_upper_actual(seg_idx) = seg_mean + margin;
    speed_lower_actual(seg_idx) = max(5, seg_mean - margin);
end

speed_upper_actual = smoothdata(speed_upper_actual, 'movmean', 20);
speed_lower_actual = smoothdata(speed_lower_actual, 'movmean', 20);

%% BUILD SEGMENT-BASED FEATURES
fprintf('\n=== CREATING SEGMENT FEATURES ===\n');

segment_distance = zeros(num_segments, 1);
segment_slope_avg = zeros(num_segments, 1);
segment_slope_max = zeros(num_segments, 1);
segment_curve_avg = zeros(num_segments, 1);
segment_curve_max = zeros(num_segments, 1);
segment_throttle_ratio = zeros(num_segments, 1);
segment_speed_upper = zeros(num_segments, 1);
segment_speed_lower = zeros(num_segments, 1);

for seg = 1:num_segments
    seg_start = (seg-1) * segment_length;
    seg_end = seg * segment_length;
    seg_idx = find(distance >= seg_start & distance < seg_end);
    
    if isempty(seg_idx)
        segment_distance(seg) = seg_start + segment_length/2;
        continue;
    end
    
    segment_distance(seg) = mean(distance(seg_idx));
    segment_slope_avg(seg) = mean(roadSlope_pct(seg_idx));
    segment_slope_max(seg) = max(abs(roadSlope_pct(seg_idx)));
    segment_curve_avg(seg) = mean(curvature(seg_idx));
    segment_curve_max(seg) = max(curvature(seg_idx));
    segment_throttle_ratio(seg) = mean(throttle_state_actual(seg_idx));
    segment_speed_upper(seg) = mean(speed_upper_actual(seg_idx));
    segment_speed_lower(seg) = mean(speed_lower_actual(seg_idx));
end

%% ESTIMATE ACTUAL RUN PARAMETERS
fprintf('\n=== ACTUAL RUN PARAMETERS ===\n');

actual_lap_time = time_s(end);
actual_energy = sum(abs(current) .* diff([time_s; time_s(end)])) * 12;
actual_max_speed = max(speed);
actual_aggressiveness = sum(throttle_state_actual) / N;

fprintf('Lap time: %.1f s\n', actual_lap_time);
fprintf('Energy: %.1f kJ\n', actual_energy/1000);
fprintf('Max speed: %.1f km/h\n', actual_max_speed);
fprintf('Aggressiveness: %.2f\n', actual_aggressiveness);

%% BUILD NARX INPUT/OUTPUT
fprintf('\n=== BUILDING NARX DATA ===\n');

% Input: 9 features per segment
U = zeros(9, num_segments);

for seg = 1:num_segments
    U(1, seg) = segment_distance(seg) / total_distance;
    U(2, seg) = segment_slope_avg(seg) / 20;
    U(3, seg) = segment_slope_max(seg) / 20;
    U(4, seg) = segment_curve_avg(seg) / 90;
    U(5, seg) = segment_curve_max(seg) / 90;
    U(6, seg) = actual_lap_time / 300;
    U(7, seg) = actual_energy / 500000;
    U(8, seg) = actual_max_speed / 50;
    U(9, seg) = actual_aggressiveness;
end

% Output: 3 features per segment
Y = zeros(3, num_segments);

for seg = 1:num_segments
    Y(1, seg) = segment_speed_upper(seg) / 50;
    Y(2, seg) = segment_speed_lower(seg) / 50;
    Y(3, seg) = segment_throttle_ratio(seg);
end

fprintf('Input U: %d x %d\n', size(U));
fprintf('Output Y: %d x %d\n', size(Y));

%% SAVE NORMALIZATION PARAMETERS
U_ps = struct();
U_ps.names = {'Position', 'SlopeAvg', 'SlopeMax', 'CurveAvg', 'CurveMax', ...
              'LapTime', 'Energy', 'MaxSpeed', 'Aggressiveness'};
U_ps.min = min(U, [], 2);
U_ps.max = max(U, [], 2);

Y_ps = struct();
Y_ps.names = {'SpeedUpper', 'SpeedLower', 'ThrottleState'};
Y_ps.min = min(Y, [], 2);
Y_ps.max = max(Y, [], 2);

%% CREATE AND TRAIN NARX
fprintf('\n=== TRAINING NARX ===\n');

Ucell = con2seq(U);
Ycell = con2seq(Y);

inputDelays = 1:3;
feedbackDelays = 1:3;
hiddenSize = 15;

net = narxnet(inputDelays, feedbackDelays, hiddenSize);
net.trainFcn = 'trainlm';
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-4;
net.trainParam.max_fail = 15;
net.trainParam.showWindow = true;

[X, Xi, Ai, T] = preparets(net, Ucell, {}, Ycell);

fprintf('Training samples: %d\n', length(X));

[net, tr] = train(net, X, T, Xi, Ai);

fprintf('\nTraining complete!\n');
fprintf('Epochs: %d\n', tr.num_epochs);
fprintf('Performance: %.6f\n', tr.best_perf);

netc = closeloop(net);

%% PREDICT OPTIMAL STRATEGY
fprintf('\n=== GENERATING OPTIMAL STRATEGY ===\n');

% Use actual parameters (or modify for different strategy)
U_optimal = U;

Ucell_opt = con2seq(U_optimal);
Ycell_init = con2seq(zeros(3, num_segments));
[~, ~, ~, Y_pred] = preparets(netc, Ucell_opt, {}, Ycell_init);

% Denormalize
Y_pred_mat = cell2mat(Y_pred);
speed_upper_pred = Y_pred_mat(1, :)' * 50;
speed_lower_pred = Y_pred_mat(2, :)' * 50;
throttle_pred = Y_pred_mat(3, :)';

% Handle size mismatch
pred_length = length(speed_upper_pred);
if pred_length < num_segments
    speed_upper_pred = [speed_upper_pred; repmat(speed_upper_pred(end), num_segments - pred_length, 1)];
    speed_lower_pred = [speed_lower_pred; repmat(speed_lower_pred(end), num_segments - pred_length, 1)];
    throttle_pred = [throttle_pred; repmat(throttle_pred(end), num_segments - pred_length, 1)];
end

%% EXTRACT THROTTLE/GLIDING START LOCATIONS FROM NARX
fprintf('\n=== EXTRACTING THROTTLE/GLIDING LOCATIONS ===\n');

% Threshold for throttle ON
throttle_binary = throttle_pred > 0.5;

% Find transitions
throttle_starts = [];
gliding_starts = [];

for seg = 2:num_segments
    if throttle_binary(seg) == 1 && throttle_binary(seg-1) == 0
        % Throttle starts
        throttle_starts = [throttle_starts; segment_distance(seg)];
    elseif throttle_binary(seg) == 0 && throttle_binary(seg-1) == 1
        % Gliding starts
        gliding_starts = [gliding_starts; segment_distance(seg)];
    end
end

fprintf('NARX Predicted Throttle Starts (m): ');
fprintf('%.1f ', throttle_starts);
fprintf('\n');
fprintf('NARX Predicted Gliding Starts (m): ');
fprintf('%.1f ', gliding_starts);
fprintf('\n');

%% STATISTICS
fprintf('\n=== STRATEGY COMPARISON ===\n');
fprintf('ACTUAL:\n');
fprintf('  Throttle events: %d\n', length(throttle_distances));
fprintf('  Avg speed window: [%.1f, %.1f] km/h\n', ...
    mean(speed_lower_actual), mean(speed_upper_actual));
fprintf('  Throttle usage: %.1f%%\n', sum(throttle_state_actual)/N*100);

fprintf('\nNARX PREDICTED:\n');
fprintf('  Throttle events: %d\n', length(throttle_starts));
fprintf('  Avg speed window: [%.1f, %.1f] km/h\n', ...
    mean(speed_lower_pred), mean(speed_upper_pred));
fprintf('  Throttle usage: %.1f%%\n', sum(throttle_binary)/num_segments*100);

%% VISUALIZATION
fprintf('\n=== CREATING VISUALIZATIONS ===\n');

figure('Position', [50, 50, 1800, 1000], 'Name', 'Strategy Comparison');

% Plot 1: Track map with actual speed
subplot(2, 4, 1);
scatter(lng, lat, 20, speed, 'filled');
colormap(gca, 'turbo');
colorbar;
title('Actual Speed');
xlabel('Longitude');
ylabel('Latitude');
grid on;
axis equal;

% Plot 2: Track features
subplot(2, 4, 2);
yyaxis left
plot(distance/1000, roadSlope_pct, 'b-', 'LineWidth', 1.5);
ylabel('Slope (%)');
yyaxis right
plot(distance/1000, curvature, 'r-', 'LineWidth', 1.5);
ylabel('Curvature (deg)');
xlabel('Distance (km)');
title('Track Features');
grid on;

% Plot 3: ACTUAL Strategy - Speed Window
subplot(2, 4, 3);
plot(distance/1000, speed_upper_actual, 'g-', 'LineWidth', 2);
hold on;
plot(distance/1000, speed_lower_actual, 'b-', 'LineWidth', 2);
plot(distance/1000, speed, 'k--', 'LineWidth', 1);
fill([distance/1000; flipud(distance/1000)], ...
     [speed_upper_actual; flipud(speed_lower_actual)], ...
     'g', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
xlabel('Distance (km)');
ylabel('Speed (km/h)');
title('ACTUAL Speed Strategy');
legend('Upper Bound', 'Lower Bound', 'Actual Speed', 'Window');
grid on;

% Plot 4: ACTUAL Strategy - Throttle Locations
subplot(2, 4, 4);
area(distance/1000, throttle_state_actual * 40, 'FaceColor', 'r', 'FaceAlpha', 0.3);
hold on;
for i = 1:length(throttle_distances)
    plot([throttle_distances(i), throttle_distances(i)]/1000, [0, 40], ...
         'g-', 'LineWidth', 2);
end
for i = 1:length(gliding_distances)
    plot([gliding_distances(i), gliding_distances(i)]/1000, [0, 40], ...
         'b-', 'LineWidth', 2);
end
xlabel('Distance (km)');
ylabel('Throttle State');
title('ACTUAL Throttle/Gliding Locations');
legend('Throttle ON', 'Throttle Start', 'Gliding Start');
grid on;
ylim([0 45]);

% Plot 5: NARX Strategy - Speed Window
subplot(2, 4, 5);
% Map segment predictions to full distance array
speed_upper_full = interp1(segment_distance, speed_upper_pred, distance, 'linear', 'extrap');
speed_lower_full = interp1(segment_distance, speed_lower_pred, distance, 'linear', 'extrap');

plot(distance/1000, speed_upper_full, 'g-', 'LineWidth', 2);
hold on;
plot(distance/1000, speed_lower_full, 'b-', 'LineWidth', 2);
plot(distance/1000, speed, 'k--', 'LineWidth', 1);
fill([distance/1000; flipud(distance/1000)], ...
     [speed_upper_full; flipud(speed_lower_full)], ...
     'g', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
xlabel('Distance (km)');
ylabel('Speed (km/h)');
title('NARX PREDICTED Speed Strategy');
legend('Upper Bound', 'Lower Bound', 'Actual Speed', 'Window');
grid on;

% Plot 6: NARX Strategy - Throttle Locations
subplot(2, 4, 6);
throttle_full = interp1(segment_distance, double(throttle_binary), distance, 'nearest', 'extrap');
area(distance/1000, throttle_full * 40, 'FaceColor', 'r', 'FaceAlpha', 0.3);
hold on;
for i = 1:length(throttle_starts)
    plot([throttle_starts(i), throttle_starts(i)]/1000, [0, 40], ...
         'g-', 'LineWidth', 2);
end
for i = 1:length(gliding_starts)
    plot([gliding_starts(i), gliding_starts(i)]/1000, [0, 40], ...
         'b-', 'LineWidth', 2);
end
xlabel('Distance (km)');
ylabel('Throttle State');
title('NARX PREDICTED Throttle/Gliding');
legend('Throttle ON', 'Throttle Start', 'Gliding Start');
grid on;
ylim([0 45]);

% Plot 7: Speed Window Comparison
subplot(2, 4, 7);
plot(distance/1000, speed_upper_actual, 'g-', 'LineWidth', 2);
hold on;
plot(distance/1000, speed_lower_actual, 'b-', 'LineWidth', 2);
plot(distance/1000, speed_upper_full, 'g--', 'LineWidth', 2);
plot(distance/1000, speed_lower_full, 'b--', 'LineWidth', 2);
xlabel('Distance (km)');
ylabel('Speed (km/h)');
title('Speed Window Comparison');
legend('Actual Upper', 'Actual Lower', 'NARX Upper', 'NARX Lower');
grid on;

% Plot 8: Throttle Comparison
subplot(2, 4, 8);
plot(distance/1000, throttle_state_actual * 100, 'r-', 'LineWidth', 2);
hold on;
plot(distance/1000, throttle_full * 100, 'b--', 'LineWidth', 2);
xlabel('Distance (km)');
ylabel('Throttle (%)');
title('Throttle Strategy Comparison');
legend('Actual', 'NARX Predicted');
grid on;

%% EXPORT FOR SIMULINK
fprintf('\n=== EXPORTING FOR SIMULINK ===\n');

simulink_data = struct();
simulink_data.time = time_s;
simulink_data.distance = distance;
simulink_data.lat = lat;
simulink_data.lng = lng;
simulink_data.speed_upper = speed_upper_full;
simulink_data.speed_lower = speed_lower_full;
simulink_data.throttle_cmd = throttle_full;
simulink_data.throttle_starts = throttle_starts;
simulink_data.gliding_starts = gliding_starts;
simulink_data.road_slope = roadSlope_pct;
simulink_data.curvature = curvature;

save(fullfile(basePath, 'optimal_strategy_for_simulink.mat'), 'simulink_data');
fprintf('Simulink data exported!\n');

%% SAVE MODEL
save(fullfile(basePath, 'NARX_SEM_Model.mat'), 'netc', 'U_ps', 'Y_ps', ...
     'segment_length', 'total_distance', 'tr');
fprintf('Model saved!\n');

fprintf('\n=== COMPLETE ===\n');
fprintf('Output file: optimal_strategy_for_simulink.mat\n');
fprintf('Contains:\n');
fprintf('  - Speed upper/lower bounds\n');
fprintf('  - Throttle command (0/1)\n');
fprintf('  - Throttle start locations\n');
fprintf('  - Gliding start locations\n');
