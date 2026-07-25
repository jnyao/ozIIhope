% Define resolutions and errors

resolutions1 = [30, 45, 90, 180];
errors1 = [2.4111154059471086e-06, 3.212194245191898e-07 , 1.018068449170344e-08, 3.1930601986111246e-10]; 

resolutions2 = [30, 45, 90, 180];
errors2 = [4.168187615312982e-06, 3.221002141669803e-07, 1.4109728296060362e-08, ...
    3.228176599914533e-10];

resolutions3 = [30, 45, 90, 180];
errors3 = [4.3749170254723816e-08, 2.6714095595682672e-09 , 2.1748608322375604e-11, 1.728118171428719e-13 ]; 

resolutions4 = [30, 45, 90, 180];
errors4 = [4.454574047617712e-08, 2.671629199040093e-09, 2.3659933687392213e-11, 1.8056979416280355e-13];

resolutions5 = [30, 45, 90, 180];
errors5 = [5.992471682193883e-10, 1.6236729283059376e-11, 5.790583778013396e-14, 4.772082900845132e-14]; 

resolutions6 = [30, 45, 90, 180];
errors6 = [ 5.999775620731451e-10, 1.8091509488455468e-11, 6.214475176173039e-14 , 5.325275276945954e-14]; 

err= [    ];

figure;
linew = 3;
loglog(resolutions1, errors1, '.-', 'DisplayName', 'FP64 5-th order', 'MarkerSize', 30, 'LineWidth', linew);
hold on;

loglog(resolutions3, errors3, '.-', 'DisplayName', 'FP64 7-th order', 'MarkerSize', 30, 'LineWidth', linew);

loglog(resolutions5, errors5, '.-', 'DisplayName', 'FP64 9-th order', 'MarkerSize', 30, 'LineWidth', linew);

loglog(resolutions2, errors2, 'o--', 'DisplayName', 'Ozaki 5-th order', 'MarkerSize', 15, 'LineWidth', linew);

loglog(resolutions4, errors4, '^--', 'DisplayName', 'Ozaki 7-th order', 'MarkerSize', 15, 'LineWidth', linew);

loglog(resolutions6, errors6, 's--', 'DisplayName', 'Ozaki 9-th order', 'MarkerSize', 15, 'LineWidth', linew);

% Add labels, legend, and grid
xlabel('Resolution (km)', 'FontSize', 18, 'FontWeight', 'bold'); % Set font size for x-axis label
ylabel('L_2 Error against Analytical Solution', 'FontSize', 18, 'FontWeight', 'bold'); % Set font size for y-axis label
title('Convergence Test', 'FontSize', 18, 'FontWeight', 'bold'); % Set font size for title

xlim([25, 200]);
legend('show', 'FontSize', 12); % Set font size for legend
ax = gca;
set(ax, 'YMinorTick', 'off'); 
set(ax, 'XMinorTick', 'off'); 
    ax.XAxis.LineWidth = 1.5;
    ax.YAxis.LineWidth = 1.5;
custom_xticks = [30, 45, 90, 180];
custom_xtick_labels = {'330', '220', '110', '55'};
xticks(custom_xticks); % Set the x-tick positions
xticklabels(custom_xtick_labels); % Set the x-tick labels
set(gca, 'FontSize', 16);
hold off;
