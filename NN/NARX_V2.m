clear; clc;
%% ========================================================================
%  NARX FOR SEM LAP STRATEGY OPTIMIZATION - PER LAP PROCESSING
%  Modified to process each lap separately based on logbook Data awal/akhir
%  FIXED: Handle empty lap_info and auto-continue laps
%% ========================================================================

%% INITIALIZATION
basePath = 'D:\.kuliah\S1\rakata\2025';
telemetryFile = fullfile(basePath, '26novfiltered.csv');
logbookFile = fullfile(basePath, 'Logbook_TD_26_10.csv');

assert(isfile(telemetryFile), 'Telemetry file not found!');
assert(isfile(logbookFile), 'Logbook file not found!');

%% LOAD TELEMETRY DATA
fprintf('=== LOADING TELEMETRY DATA ===\n');
T = readtable(telemetryFile);

lat_all = double(T.lat);
lng_all = double(T.lng);
speed_all = double(T.kecepatan);
throttle_all = double(T.throttle);
current_all = double(T.arus);
millis_all = double(T.millis);

% Remove invalid GPS (0,0)
valid = ~(lat_all == 0 & lng_all == 0);
lat_all = lat_all(valid);
lng_all = lng_all(valid);
speed_all = speed_all(valid);
throttle_all = throttle_all(valid);
current_all = current_all(valid);
millis_all = millis_all(valid);

fprintf('Total telemetry points: %d\n', length(lat_all));

%% LOAD LOGBOOK
fprintf('\n=== LOADING LOGBOOK ===\n');
LB = readtable(logbookFile);

% Get column names
throttleCols = contains(lower(LB.Properties.VariableNames), 'throtle') & ...
               contains(lower(LB.Properties.VariableNames), '_s');
glidingCols = contains(lower(LB.Properties.VariableNames), 'gliding') & ...
              contains(lower(LB.Properties.VariableNames), '_s');

throttleVars = LB.Properties.VariableNames(throttleCols);
glidingVars = LB.Properties.VariableNames(glidingCols);

% Check if Data awal and Data akhir columns exist
if ~any(contains(LB.Properties.VariableNames, 'Data_awal', 'IgnoreCase', true))
    error('Column "Data awal" not found in logbook!');
end
if ~any(contains(LB.Properties.VariableNames, 'Data_akhir', 'IgnoreCase', true))
    error('Column "Data akhir" not found in logbook!');
end

% Get column names (handle different naming conventions)
dataAwalCol = LB.Properties.VariableNames(contains(LB.Properties.VariableNames, 'Data_awal', 'IgnoreCase', true));
dataAkhirCol = LB.Properties.VariableNames(contains(LB.Properties.VariableNames, 'Data_akhir', 'IgnoreCase', true));

dataAwalCol = dataAwalCol{1};
dataAkhirCol = dataAkhirCol{1};

num_laps = height(LB);
fprintf('Number of laps in logbook: %d\n', num_laps);

%% AUTO-FILL MISSING LAP BOUNDARIES
% Jika Data awal kosong tapi lap sebelumnya ada Data akhir, gunakan Data akhir sebagai Data awal
% Jika Data akhir kosong, gunakan Data awal lap berikutnya (atau akhir data)
fprintf('\n=== AUTO-FILLING MISSING LAP BOUNDARIES ===\n');

for lap_idx = 1:num_laps
    data_awal = LB.(dataAwalCol)(lap_idx);
    data_akhir = LB.(dataAkhirCol)(lap_idx);
    
    % Jika Data awal kosong, ambil dari Data akhir lap sebelumnya
    if isnan(data_awal) && lap_idx > 1
        prev_akhir = LB.(dataAkhirCol)(lap_idx - 1);
        if ~isnan(prev_akhir)
            LB.(dataAwalCol)(lap_idx) = prev_akhir;
            fprintf('Lap %d: Data awal filled from previous lap akhir: %.0f\n', lap_idx, prev_akhir);
        end
    end
    
    % Jika Data akhir kosong, ambil dari Data awal lap berikutnya
    if isnan(data_akhir)
        if lap_idx < num_laps
            next_awal = LB.(dataAwalCol)(lap_idx + 1);
            if ~isnan(next_awal)
                LB.(dataAkhirCol)(lap_idx) = next_awal;
                fprintf('Lap %d: Data akhir filled from next lap awal: %.0f\n', lap_idx, next_awal);
            end
        else
            % Lap terakhir, gunakan akhir data
            LB.(dataAkhirCol)(lap_idx) = length(lat_all);
            fprintf('Lap %d: Data akhir filled with end of data: %d\n', lap_idx, length(lat_all));
        end
    end
end

%% INITIALIZE STORAGE FOR ALL LAPS
all_laps_U = [];
all_laps_Y = [];
lap_info = [];  % Changed to empty array instead of struct()
valid_lap_count = 0;

%% PROCESS EACH LAP
for lap_idx = 1:num_laps
    fprintf('\n========================================\n');
    fprintf('PROCESSING LAP %d / %d\n', lap_idx, num_laps);
    fprintf('========================================\n');
    
    % Get lap boundaries from logbook
    data_awal = LB.(dataAwalCol)(lap_idx);
    data_akhir = LB.(dataAkhirCol)(lap_idx);
    
    % Skip if boundaries are invalid
    if isnan(data_awal) || isnan(data_akhir) || data_awal >= data_akhir
        fprintf('Invalid lap boundaries. Skipping lap %d.\n', lap_idx);
        continue;
    end
    
    % Extract data for this lap based on row indices
    lap_start_idx = max(1, round(data_awal));
    lap_end_idx = min(length(lat_all), round(data_akhir));
    
    fprintf('Lap data range: row %d to %d\n', lap_start_idx, lap_end_idx);
    
    % Extract lap data
    lat = lat_all(lap_start_idx:lap_end_idx);
    lng = lng_all(lap_start_idx:lap_end_idx);
    speed = speed_all(lap_start_idx:lap_end_idx);
    throttle = throttle_all(lap_start_idx:lap_end_idx);
    current = current_all(lap_start_idx:lap_end_idx);
    millis = millis_all(lap_start_idx:lap_end_idx);
    
    N = length(lat);
    fprintf('Lap %d data points: %d\n', lap_idx, N);
    
    if N < 10
        fprintf('Insufficient data points. Skipping lap %d.\n', lap_idx);
        continue;
    end
    
    %% GPS FILTERING FOR THIS LAP - LESS AGGRESSIVE
    % Hitung bounding box dari data aktual dengan margin
    lat_center = median(lat);
    lng_center = median(lng);
    lat_range = max(lat) - min(lat);
    lng_range = max(lng) - min(lng);
    
    LAT_MIN = lat_center - lat_range * 2;
    LAT_MAX = lat_center + lat_range * 2;
    LNG_MIN = lng_center - lng_range * 2;
    LNG_MAX = lng_center + lng_range * 2;
    MAX_JUMP_DISTANCE = 500; % Dikurangi dari 1000m jadi 500m
    MIN_DISTANCE = 0.01; % Dikurangi untuk data lebih detail
    
    haversine = @(lat1, lon1, lat2, lon2) ...
        6371000 * 2 * asin(sqrt(sind((lat2-lat1)/2).^2 + ...
        cosd(lat1) .* cosd(lat2) .* sind((lon2-lon1)/2).^2));
    
    is_valid = true(N, 1);
    
    % Filter 1: Geographic bounds (lebih permisif)
    out_of_bounds = lat < LAT_MIN | lat > LAT_MAX | ...
                    lng < LNG_MIN | lng > LNG_MAX;
    is_valid = is_valid & ~out_of_bounds;
    
    % Filter 2: Distance jumps (hanya extreme outliers)
    if N > 1
        distances = zeros(N-1, 1);
        for i = 1:N-1
            distances(i) = haversine(lat(i), lng(i), lat(i+1), lng(i+1));
        end
        
        % Hanya filter jump yang sangat besar
        for i = 1:N-1
            if distances(i) > MAX_JUMP_DISTANCE
                is_valid(i) = false;
                is_valid(i+1) = false;
            end
        end
    end
    
    % Apply filter
    lat = lat(is_valid);
    lng = lng(is_valid);
    speed = speed(is_valid);
    throttle = throttle(is_valid);
    current = current(is_valid);
    millis = millis(is_valid);
    
    N = length(lat);
    fprintf('After filtering: %d points\n', N);
    
    if N < 10
        fprintf('Insufficient data after filtering. Skipping lap %d.\n', lap_idx);
        continue;
    end
    
    %% CALCULATE DISTANCE
    R = 6371000;
    dlat = deg2rad(diff(lat));
    dlng = deg2rad(diff(lng));
    a = sin(dlat/2).^2 + cos(deg2rad(lat(1:end-1))) .* ...
        cos(deg2rad(lat(2:end))) .* sin(dlng/2).^2;
    d = 2 * R * atan2(sqrt(a), sqrt(1-a));
    d(d < MIN_DISTANCE) = MIN_DISTANCE;
    distance = [0; cumsum(d)];
    
    total_distance = distance(end);
    fprintf('Total distance: %.2f m\n', total_distance);
    
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
    curvature = smoothdata(curvature, 'gaussian', min(10, ceil(N/10)));
    
    %% GET THROTTLE/GLIDING EVENTS FOR THIS LAP
    time_s = (millis - millis(1)) / 1000;
    
    % Extract events from logbook
    eventThrottle = [];
    eventGliding = [];
    
    for col_idx = 1:length(throttleVars)
        val = LB.(throttleVars{col_idx})(lap_idx);
        if ~isnan(val) && val > 0
            eventThrottle = [eventThrottle; val];
        end
    end
    
    for col_idx = 1:length(glidingVars)
        val = LB.(glidingVars{col_idx})(lap_idx);
        if ~isnan(val) && val > 0
            eventGliding = [eventGliding; val];
        end
    end
    
    fprintf('Throttle events: %d\n', length(eventThrottle));
    fprintf('Gliding events: %d\n', length(eventGliding));
    
    %% MAP EVENTS TO DISTANCE
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
    
    %% CREATE THROTTLE STATE
    throttle_state_actual = zeros(N, 1);
    
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
    
    fprintf('Throttle ON: %.1f%% of lap\n', sum(throttle_state_actual)/N*100);
    
    %% CALCULATE SPEED WINDOWS - IMPROVED METHOD
    % Gunakan percentile-based windowing yang lebih smooth
    segment_length = 50; % Dikurangi dari 100m untuk resolusi lebih tinggi
    num_segments = max(1, ceil(total_distance / segment_length));
    
    speed_upper_actual = zeros(N, 1);
    speed_lower_actual = zeros(N, 1);
    
    % Window size untuk smoothing (dalam meter)
    window_size = 150; % 150m rolling window
    
    for i = 1:N
        % Ambil data dalam radius window_size
        in_window = abs(distance - distance(i)) <= window_size/2;
        
        if sum(in_window) < 3
            in_window = true(size(distance)); % Fallback ke semua data
        end
        
        window_speeds = speed(in_window);
        
        % Gunakan percentile untuk robustness
        speed_p75 = prctile(window_speeds, 75);
        speed_p25 = prctile(window_speeds, 25);
        speed_median = median(window_speeds);
        
        % IQR-based margin
        iqr_margin = (speed_p75 - speed_p25) * 1.5;
        iqr_margin = max(iqr_margin, 5); % Minimum 5 km/h margin
        
        speed_upper_actual(i) = min(50, speed_median + iqr_margin);
        speed_lower_actual(i) = max(3, speed_median - iqr_margin);
    end
    
    % Apply additional smoothing
    speed_upper_actual = smoothdata(speed_upper_actual, 'gaussian', min(50, ceil(N/5)));
    speed_lower_actual = smoothdata(speed_lower_actual, 'gaussian', min(50, ceil(N/5)));
    
    %% BUILD SEGMENT FEATURES
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
    
    %% CALCULATE LAP PARAMETERS
    lap_time = time_s(end);
    lap_energy = sum(abs(current) .* diff([time_s; time_s(end)])) * 12;
    lap_max_speed = max(speed);
    lap_aggressiveness = sum(throttle_state_actual) / N;
    
    fprintf('Lap time: %.1f s\n', lap_time);
    fprintf('Energy: %.1f kJ\n', lap_energy/1000);
    fprintf('Max speed: %.1f km/h\n', lap_max_speed);
    fprintf('Aggressiveness: %.2f\n', lap_aggressiveness);
    
    %% BUILD NARX INPUT/OUTPUT FOR THIS LAP
    U_lap = zeros(9, num_segments);
    
    for seg = 1:num_segments
        U_lap(1, seg) = segment_distance(seg) / max(total_distance, 1);
        U_lap(2, seg) = segment_slope_avg(seg) / 20;
        U_lap(3, seg) = segment_slope_max(seg) / 20;
        U_lap(4, seg) = segment_curve_avg(seg) / 90;
        U_lap(5, seg) = segment_curve_max(seg) / 90;
        U_lap(6, seg) = lap_time / 300;
        U_lap(7, seg) = lap_energy / 500000;
        U_lap(8, seg) = lap_max_speed / 50;
        U_lap(9, seg) = lap_aggressiveness;
    end
    
    Y_lap = zeros(3, num_segments);
    
    for seg = 1:num_segments
        Y_lap(1, seg) = segment_speed_upper(seg) / 50;
        Y_lap(2, seg) = segment_speed_lower(seg) / 50;
        Y_lap(3, seg) = segment_throttle_ratio(seg);
    end
    
    %% ACCUMULATE DATA
    all_laps_U = [all_laps_U, U_lap];
    all_laps_Y = [all_laps_Y, Y_lap];
    
    % Store lap information - FIXED: use array indexing
    valid_lap_count = valid_lap_count + 1;
    lap_info(valid_lap_count).lap_number = lap_idx;
    lap_info(valid_lap_count).num_segments = num_segments;
    lap_info(valid_lap_count).total_distance = total_distance;
    lap_info(valid_lap_count).lap_time = lap_time;
    lap_info(valid_lap_count).lap_energy = lap_energy;
    lap_info(valid_lap_count).throttle_distances = throttle_distances;
    lap_info(valid_lap_count).gliding_distances = gliding_distances;
    lap_info(valid_lap_count).distance = distance;
    lap_info(valid_lap_count).speed = speed;
    lap_info(valid_lap_count).speed_upper = speed_upper_actual;
    lap_info(valid_lap_count).speed_lower = speed_lower_actual;
    lap_info(valid_lap_count).throttle_state = throttle_state_actual;
    lap_info(valid_lap_count).lat = lat;
    lap_info(valid_lap_count).lng = lng;
    
    fprintf('Lap %d processed: %d segments\n', lap_idx, num_segments);
end

%% CHECK IF WE HAVE DATA
if isempty(all_laps_U)
    error('No valid lap data found! Check logbook Data awal/akhir values.');
end

fprintf('\n=== TOTAL ACCUMULATED DATA ===\n');
fprintf('Total segments across all laps: %d\n', size(all_laps_U, 2));
fprintf('Valid laps processed: %d\n', valid_lap_count);

%% SAVE NORMALIZATION PARAMETERS
U_ps = struct();
U_ps.names = {'Position', 'SlopeAvg', 'SlopeMax', 'CurveAvg', 'CurveMax', ...
              'LapTime', 'Energy', 'MaxSpeed', 'Aggressiveness'};
U_ps.min = min(all_laps_U, [], 2);
U_ps.max = max(all_laps_U, [], 2);

Y_ps = struct();
Y_ps.names = {'SpeedUpper', 'SpeedLower', 'ThrottleState'};
Y_ps.min = min(all_laps_Y, [], 2);
Y_ps.max = max(all_laps_Y, [], 2);

%% CREATE AND TRAIN NARX
fprintf('\n=== TRAINING NARX WITH MULTI-LAP DATA ===\n');

Ucell = con2seq(all_laps_U);
Ycell = con2seq(all_laps_Y);

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

%% SAVE MODEL AND DATA
save(fullfile(basePath, 'NARX_SEM_Model_PerLap.mat'), 'netc', 'U_ps', 'Y_ps', ...
     'lap_info', 'all_laps_U', 'all_laps_Y', 'tr');
fprintf('\nModel saved: NARX_SEM_Model_PerLap.mat\n');

%% VISUALIZATION - IMPROVED LAYOUT
fprintf('\n=== CREATING VISUALIZATION FOR LAP 1 ===\n');

if ~isempty(lap_info) && length(lap_info) >= 1
    lap1 = lap_info(1);
    
    figure('Position', [50, 50, 1800, 1000], 'Name', sprintf('Lap %d Strategy Analysis', lap1.lap_number));
    
    % Plot 1: GPS Track dengan Throttle Strategy Overlay
    subplot(2, 3, 1);
    % Buat colormap custom: merah untuk throttle ON, biru untuk gliding
    colors = zeros(length(lap1.speed), 3);
    throttle_on = lap1.throttle_state > 0.5;
    colors(throttle_on, 1) = 1; % Red channel untuk throttle
    colors(~throttle_on, 3) = 0.7; % Blue channel untuk gliding
    
    scatter(lap1.lng, lap1.lat, 30, colors, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    % Tambahkan marker untuk throttle start dan gliding start
    hold on;
    for i = 1:length(lap1.throttle_distances)
        [~, idx] = min(abs(lap1.distance - lap1.throttle_distances(i)));
        plot(lap1.lng(idx), lap1.lat(idx), 'go', 'MarkerSize', 12, 'LineWidth', 3);
    end
    for i = 1:length(lap1.gliding_distances)
        [~, idx] = min(abs(lap1.distance - lap1.gliding_distances(i)));
        plot(lap1.lng(idx), lap1.lat(idx), 'ms', 'MarkerSize', 12, 'LineWidth', 3);
    end
    
    title(sprintf('Lap %d - GPS Track + Throttle Strategy', lap1.lap_number));
    xlabel('Longitude');
    ylabel('Latitude');
    legend('Track', 'Throttle Start', 'Gliding Start', 'Location', 'best');
    grid on;
    axis equal tight;
    
    % Plot 2: Speed Window (Improved)
    subplot(2, 3, 2);
    % Plot dengan transparansi lebih baik
    h1 = plot(lap1.distance/1000, lap1.speed_upper, 'g-', 'LineWidth', 2.5);
    hold on;
    h2 = plot(lap1.distance/1000, lap1.speed_lower, 'b-', 'LineWidth', 2.5);
    h3 = plot(lap1.distance/1000, lap1.speed, 'k-', 'LineWidth', 1.5);
    
    % Fill area dengan gradient
    fill([lap1.distance/1000; flipud(lap1.distance/1000)], ...
         [lap1.speed_upper; flipud(lap1.speed_lower)], ...
         'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    xlabel('Distance (km)');
    ylabel('Speed (km/h)');
    title('Speed Strategy Window (Optimized)');
    legend([h1 h2 h3], {'Upper Bound', 'Lower Bound', 'Actual Speed'}, 'Location', 'best');
    grid on;
    xlim([0 max(lap1.distance/1000)]);
    ylim([0 max(lap1.speed_upper)*1.1]);
    
    % Plot 3: Speed colored by Throttle State
    subplot(2, 3, 3);
    scatter(lap1.distance/1000, lap1.speed, 30, lap1.throttle_state, 'filled');
    colormap(gca, [0 0.4 0.8; 1 0.2 0.2]); % Blue untuk gliding, Red untuk throttle
    colorbar('Ticks', [0, 1], 'TickLabels', {'Gliding', 'Throttle'});
    xlabel('Distance (km)');
    ylabel('Speed (km/h)');
    title('Speed vs Distance (colored by Throttle State)');
    grid on;
    xlim([0 max(lap1.distance/1000)]);
    
    % Plot 4-6: Show statistics for all laps
    subplot(2, 3, 4);
    if ~isempty(lap_info)
        lap_numbers = arrayfun(@(x) x.lap_number, lap_info);
        lap_times = arrayfun(@(x) x.lap_time, lap_info);
        bar(lap_numbers, lap_times, 'FaceColor', [0.2 0.6 0.8]);
        xlabel('Lap Number');
        ylabel('Time (s)');
        title('Lap Times');
        grid on;
        
        % Tambahkan value labels
        for i = 1:length(lap_times)
            text(lap_numbers(i), lap_times(i), sprintf('%.0fs', lap_times(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    
    subplot(2, 3, 5);
    if ~isempty(lap_info)
        lap_energies = arrayfun(@(x) x.lap_energy/1000, lap_info);
        bar(lap_numbers, lap_energies, 'FaceColor', [0.8 0.4 0.2]);
        xlabel('Lap Number');
        ylabel('Energy (kJ)');
        title('Energy Consumption per Lap');
        grid on;
        
        % Tambahkan value labels
        for i = 1:length(lap_energies)
            text(lap_numbers(i), lap_energies(i), sprintf('%.1fkJ', lap_energies(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    
    subplot(2, 3, 6);
    if ~isempty(lap_info)
        lap_distances = arrayfun(@(x) x.total_distance/1000, lap_info);
        bar(lap_numbers, lap_distances, 'FaceColor', [0.4 0.8 0.4]);
        xlabel('Lap Number');
        ylabel('Distance (km)');
        title('Lap Distances');
        grid on;
        
        % Tambahkan value labels
        for i = 1:length(lap_distances)
            text(lap_numbers(i), lap_distances(i), sprintf('%.2fkm', lap_distances(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    
    % Tambahkan suptitle dengan info lap
    sgtitle(sprintf('Lap %d Analysis | Distance: %.2fkm | Time: %.1fs | Energy: %.1fkJ', ...
            lap1.lap_number, lap1.total_distance/1000, lap1.lap_time, lap1.lap_energy/1000), ...
            'FontSize', 14, 'FontWeight', 'bold');
end

fprintf('\n=== PROCESSING COMPLETE ===\n');
fprintf('Processed %d laps successfully\n', length(lap_info));
fprintf('Model trained with %d total segments\n', size(all_laps_U, 2));
