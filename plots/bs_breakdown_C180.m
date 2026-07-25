function bs_breakdown_C180()
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
        42.215252161026,  5.931260108947754, 9.805611848831177 ;    
        42.21429204940796, 6.874910831451416,  11.817328691482544;   
        42.25006985664368, 8.366849660873413,  13.918232440948486;   
        42.273446798324585, 10.350630521774292, 16.002113819122314    
    ];
    data1 = round(data1, 2);
    
    data2 = [
        22.583706855773926,  5.902933120727539, 9.856273651123047 ;    
        22.59729814529419, 6.951570987701416,  11.918834447860718;   
        22.596157789230347, 8.423571586608887,  13.975680112838745;   
        40.80984401702881, 10.507200002670288, 16.089783191680908 
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
            
            if i == 4
                plot(x_offset + i, total + 9, 'rv', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
                text(x_offset + i, total + 12, {'required', 'numsplit'}, ...
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
    
    title('Performance Breakdown of Ozaki Scheme (7th-order, 55 km)', 'FontSize', 21, 'FontWeight', 'bold');
    
    legend(bar_handles(1:length(stack_labels)), stack_labels, 'Location', 'northwest', 'FontSize', 15);
    
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
    
    
legend(bar_handles(1:length(stack_labels)), stack_labels, 'Location', 'northwest', 'FontSize', 14);
    
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