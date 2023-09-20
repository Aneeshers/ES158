%% adapted from https://github.com/ComputationalRobotics/OptimalControlEstimation-Examples/blob/main/grid_world_value_iteration.m
%%
clc; clear; close all;

%% Create problem data
% States
[X,Y] = meshgrid(1:10,1:10);
nx = 10; % number of states in one dimension

% Controls
U = [1, 0;
     -1, 0;
     0, 1;
     0, -1;
     0, 0];
nu = size(U,1); % number of controls

% Create vector g
target = [1,10]; 
obstacles = [2,2;
             2,3;
             3,2;
             3,3;
             3,4;
             4,5;
             4,6;
             5,5;
             5,6;
             6,5;
             6,6;
             7,7;
             7,8;
             8,2;
             8,3;
             8,4;
             8,7;
             8,8;
             9,7;
             9,8;
             10,7;
             10,8];

g = zeros(10, 10, 5);
for i1 = 1:10
    for i2 = 1:10
        x = [i1, i2];
        
        if ismember(x, target, 'rows')
            state_cost = 0;
        elseif ismember(x, obstacles, 'rows')
            state_cost = 20;
        else
            state_cost = 1;
        end

        for u = 1:5
            g(i1, i2, u) = state_cost;
        end
    end
end

%% Construct Transition Tensor P
P = zeros(10, 10, 5, 10, 10);

for j = 1:5 % over controls
    u = U(j,:); % control
    for i1 = 1:10
        for i2 = 1:10
            x = [i1, i2];
            xp = x + u; % next state
            
            if xp(1) < 1 || xp(1) > 10 || xp(2) < 1 || xp(2) > 10 % check if next state is outside the grid
                P(i1, i2, j, i1, i2) = 1; % stay in the current state
            else
                P(i1, i2, j, xp(1), xp(2)) = 1; % transition to the next state
            end
        end
    end
end
Q = zeros(10, 10, 5);
MAX_ITERS = 1e3;
Q_diff_values = [];

%% Start value iteration using tensors

gif_filename = 'value_iteration.gif';

for iter = 1:MAX_ITERS
    Q_prev = Q; % Store previous Q values to check convergence
    
    for u = 1:5 % over controls
        for i1 = 1:10
            for i2 = 1:10
                next_values = squeeze(P(i1, i2, u, :, :)) .* min(Q, [], 3);
                Q_new(i1, i2, u) = g(i1, i2, u) + sum(sum(next_values));
            end
        end
    end

    Q_diff = norm(Q_new(:) - Q_prev(:));
    Q_diff_values = [Q_diff_values; Q_diff];
    Q = Q_new;
    
    % Visualization
    J_mat_current = zeros(10,10);
    for u = 1:5
        for i1 = 1:10
            for i2 = 1:10
                J_mat_current(i1, i2) = min(Q(i1, i2, :));
            end
        end
    end
    clf;
    image(J_mat_current,'CDataMapping','scaled'); colorbar;
    set(gca,'XAxisLocation','top')
    title(['Value Iteration: Iteration ' num2str(iter)])
    drawnow;
    
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    
    if iter == 1
        imwrite(imind,cm,gif_filename,'gif', 'Loopcount',inf, 'DelayTime',0.1);
    else
        imwrite(imind,cm,gif_filename,'gif','WriteMode','append', 'DelayTime',0.1);
    end
    
    % Check convergence
    if Q_diff < 1e-6
        fprintf("Value iteration converged in %d iterations, Q_diff = %3.2e.\n",iter,Q_diff)
        break
    end
end


%% C-TRIAL:

initial_values = [0, 4, 8, 12]; % list of initial c values
convergence_iters = zeros(length(initial_values), 1);
final_J_values = cell(length(initial_values), 1);

for idx = 1:length(initial_values)
    c = initial_values(idx);
    Q = c * ones(10, 10, 5); % Initialize Q with the current constant c

    % Rest of the value iteration loop remains the same as before...

    for iter = 1:MAX_ITERS
        Q_prev = Q; % Store previous Q values before updating

        for u = 1:5 % over controls
            for i1 = 1:10
                for i2 = 1:10
                    next_values = squeeze(P(i1, i2, u, :, :)) .* min(Q, [], 3);
                    Q_new(i1, i2, u) = g(i1, i2, u) + sum(sum(next_values));
                end
            end
        end
    
        Q_diff = norm(Q_new(:) - Q_prev(:));
        Q_diff_values = [Q_diff_values; Q_diff];
        Q = Q_new;
        
        % Visualization
        J_mat_current = zeros(10,10);
        for u = 1:5
            for i1 = 1:10
                for i2 = 1:10
                    J_mat_current(i1, i2) = min(Q(i1, i2, :));
                end
            end
        end
        clf;
        image(J_mat_current,'CDataMapping','scaled'); colorbar;
        set(gca,'XAxisLocation','top')
        title(['Value Iteration: Iteration ' num2str(iter)])
        pause(0.1);

        % Check convergence
        if Q_diff < 1e-6
            fprintf("Value iteration with initialization %d converged in %d iterations, Q_diff = %3.2e.\n", c, iter, Q_diff);
            convergence_iters(idx) = iter;
            break
        end
    end

    final_J_values{idx} = min(Q, [], 3);
end

% results
for idx = 1:length(initial_values)
    fprintf("Initialization = %d, Iterations = %d\n", initial_values(idx), convergence_iters(idx));
end
figure;

for idx = 1:length(initial_values)
    subplot(2, 2, idx); 
    imagesc(final_J_values{idx});
    colorbar;
    
    max_val = max(max(final_J_values{idx}));
    
    % Overlay the max value on the plot
    text(5, 5, num2str(max_val), 'FontSize', 12, 'HorizontalAlignment', 'center', 'Color', 'r');
    
    title(['Initialization: ', num2str(initial_values(idx))]);
    set(gca,'XAxisLocation','top');
end

%% SOR
%% Create problem data for a larger grid
GRID_SIZE = 50;

% States
[X,Y] = meshgrid(1:GRID_SIZE,1:GRID_SIZE);
X = X(:);
Y = Y(:);
XY = [X,Y];
nx = length(X); % number of states

% Controls (remain the same)
U = [1, 0;
     -1, 0;
     0, 1;
     0, -1;
     0, 0];
nu = size(U,1); % number of controls

target = [1, GRID_SIZE];

obstacles = [];
for i = 3:4:GRID_SIZE-2
    obstacles = [obstacles; 
                 i*ones(GRID_SIZE-2, 1), (2:GRID_SIZE-1)'];
    obstacles = [obstacles; 
                 (i+1)*ones(GRID_SIZE-2, 1), flipud(2:GRID_SIZE-1)'];
end

g = zeros(GRID_SIZE, GRID_SIZE, nu);
for i = 1:nx
    for j = 1:nu
        x = XY(i,:); % state
        if ismember(x, target, 'rows')
            state_cost = 0;
        elseif ismember(x, obstacles, 'rows')
            state_cost = 20;
        else
            state_cost = 1;
        end

        u = U(j,:); % control
        control_cost = sum(u.^2); 

        g(x(1), x(2), j) = state_cost; % Assign the cost to the correct position in g
    end
end


%% Construct Transition Tensor P
P = zeros(GRID_SIZE, GRID_SIZE, nu, GRID_SIZE, GRID_SIZE);

for i1 = 1:GRID_SIZE
    for i2 = 1:GRID_SIZE
        for u = 1:nu
            % Compute next state based on current state and control
            x_next = [i1, i2] + U(u, :);

            % Check if next state is within grid boundaries
            if x_next(1) >= 1 && x_next(1) <= GRID_SIZE && x_next(2) >= 1 && x_next(2) <= GRID_SIZE
                P(i1, i2, u, x_next(1), x_next(2)) = 1;
            else
                % If next state is outside the grid, stay in the current state
                P(i1, i2, u, i1, i2) = 1;
            end
        end
    end
end

Q = zeros(GRID_SIZE, GRID_SIZE, nu);
MAX_ITERS = 1e3;
Q_diff_values = [];

% range of omega values to test
omega_values = [1.1, 1.8, 2.5, 3.7, 4.9];
convergence_iters_sor = zeros(length(omega_values), 1);
final_J_values_sor = cell(length(omega_values), 1);

for idx = 1:length(omega_values)
    omega = omega_values(idx);
    Q = zeros(GRID_SIZE, GRID_SIZE, nu);

    % Value iteration loop with SOR
    for iter = 1:MAX_ITERS
        Q_prev = Q; % Store previous Q values before updating

        for u = 1:5 % over controls
            for i1 = 1:GRID_SIZE
                for i2 = 1:GRID_SIZE
                    next_values = squeeze(P(i1, i2, u, :, :)) .* min(Q, [], 3);
                    Q_new(i1, i2, u) = g(i1, i2, u) + sum(sum(next_values));
                end
            end
        end

        Q_diff = norm(Q_new(:) - Q_prev(:));
        Q_diff_values = [Q_diff_values; Q_diff];
        Q = Q_new;
    
        % Apply SOR
        Q = (1 - omega) * Q + omega * Q_new;
                

        Q_diff = norm(Q - Q_prev, 'fro');
        if Q_diff < 1e-6
            fprintf("Value iteration with SOR (omega = %f) converged in %d iterations, Q_diff = %3.2e.\n", omega, iter, Q_diff);
            convergence_iters_sor(idx) = iter;
            break
        end
    end

    final_J_values_sor{idx} = min(Q, [], 3);
end


% Visualization
figure;
for idx = 1:length(omega_values)
    subplot(2, 3, idx);
    imagesc(final_J_values_sor{idx});
    colorbar;
    max_val = max(max(final_J_values_sor{idx}));
    text(5, 5, num2str(max_val), 'FontSize', 12, 'HorizontalAlignment', 'center', 'Color', 'r');
    title(['SOR (omega = ', num2str(omega_values(idx)), ')']);
    set(gca,'XAxisLocation','top');
end


