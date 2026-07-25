function bs_breakdown_fp16()
    numsplits = {'in-chnl utilization (numsplit n)', 'batch utilization (numsplit n)'};
    categories = {'4', '5', '6', '7'};
    stack_labels = {'computation', 'split', 'FP64 accumulation'};
    
    data2 = [
        17.293, 3.719,  3.286;   
        51.889, 3.734,  4.443;  
        103.8, 4.283,  4.721;   
        173.014, 5.554, 6.218   
    ];

    data1 = [
        6.443228006362915,  2.465571880340576, 1.8735401630401611 ;    
        6.454065322875977, 2.8290469646453857,  2.1568517684936523;   
        6.461787462234497, 3.134037494659424,  2.436161518096924;   
        6.473891258239746, 3.5368666648864746, 2.731147289276123    
    ];
    data1 = round(data1, 2);
    
    data2 = [
        3.5644562244415283,  2.8394131660461426, 1.913336992263794 ;    
        3.5623984336853027, 3.1280341148376465,  2.177424192428589;   
        3.5629184246063232, 3.42995023727417,  2.4588983058929443;   
        6.0804290771484375, 3.692142963409424, 2.732581615447998 
    ];
    data2 = round(data2, 2);
    
    all_data = {data1, data2};
    
    figure('Position', [146,536,720,343], 'Color', 'white');

    colors = [
        0.2, 0.4, 0.8;  
        0.8, 0.4, 0.2;  
        0.4, 0.8, 0.4  
    ];
    
    n_splits = length(numsplits);
    n_categories = length(categories);
    
    split_spacing = 1;
    
    x_positions = [];
    n_centers = [];
    
    for nidx = 1:n_splits
        split_center = (nidx - 1) * (n_categories + split_spacing) + (n_categories + 1) / 2;
        n_centers = [n_centers, split_center];
        
        for cat_idx = 1:n_categories
            x_pos = (nidx - 1) * (n_categories + split_spacing) + cat_idx;
            x_positions = [x_positions, x_pos];
        end
    end
    
    hold on;
    
    bar_handles = [];
    
    for nidx = 1:n_splits
        numsplit_data = all_data{nidx};
        
        x_offset = (nidx - 1) * (n_categories + split_spacing);
        
        h = bar(x_offset + (1:n_categories), numsplit_data, 'stacked');
        
        for i = 1:length(h)
            h(i).FaceColor = colors(i, :);
            h(i).EdgeColor = 'white';
            h(i).LineWidth = 1;
        end
        
        bar_handles = [bar_handles, h];
        
        for i = 1:size(numsplit_data, 1)
            cumulative_height = 0;
            for j = 1:size(numsplit_data, 2)
                value = numsplit_data(i, j);
                cumulative_height = cumulative_height + value/2;
                text(x_offset + i, cumulative_height, num2str(value), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 15, ...
                    'FontWeight', 'bold', ...
                    'Color', 'white');
                cumulative_height = cumulative_height + value/2;
            end
            
            total = sum(numsplit_data(i, :));
            text(x_offset + i, total + 0.05, [num2str(total)], ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
            if i == 3
                plot(x_offset + i, total + 1.6, 'rv', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
                text(x_offset + i, total + 2, {'required', 'numsplit'}, ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'bottom', ...
                    'FontSize', 14, ...
                    'FontWeight', 'bold',...
                    'Color', 'r');
            end
            
        end
    end
    set(gca, 'XTick', x_positions);
    
    x_labels = [categories];
    set(gca, 'XTickLabel', x_labels, 'FontSize', 13);
    
    for i = 1:length(n_centers)
        text(n_centers(i), -max(ylim)*0.08, numsplits{i}, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 15, ...
            'FontWeight', 'bold');
    end
    
    ylabel('Runtime (s)', 'FontSize', 20, 'FontWeight', 'bold');
    
    title('Performance Breakdown of Ozaki Scheme (7th-order, 110 km)', 'FontSize', 21, 'FontWeight', 'bold');
    
    legend(bar_handles(1:length(stack_labels)), stack_labels, 'Location', 'northeast', 'FontSize', 15);
    
    grid on;
    
    y_max = 0;
    for nidx = 1:n_splits
        numsplit_data = all_data{nidx};
        for i = 1:size(numsplit_data, 1)
            total = sum(numsplit_data(i, :));
            y_max = max(y_max, total);
        end
    end
    ylim([0, y_max * 1.4]);
    
    trend_colors = [
        0.8, 0.2, 0.2;  
        0.2, 0.8, 0.2;  
        0.2, 0.2, 0.8;  
        0.8, 0.8, 0.2   
    ];
    
    
legend(bar_handles(1:length(stack_labels)), stack_labels, 'Location', 'northeast', 'FontSize', 14);
    
    set(gcf, 'Color', 'w');
    box on;
    
    yPosition = 3.05; 
    lineStyle = '--'; 
    lineColor = 'r'; 
    lineWidth = 2;
    xLimits = xlim;
    yLimits = ylim;
    
    
    hold off;
end
