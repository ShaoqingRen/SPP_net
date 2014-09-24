function showboxes_new(im, boxes, legends)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes)
%
% -------------------------------------------------------

fix_width = 800;
imsz = size(im);
scale = fix_width / imsz(2);
im = imresize(im, scale);

boxes = cellfun(@(x) x * scale, boxes, 'UniformOutput', false);

image(im); 
axis image;
axis off;
set(gcf, 'Color', 'white');

valid_boxes = cellfun(@(x) ~isempty(x), boxes, 'UniformOutput', true);
valid_boxes_num = sum(valid_boxes);

if valid_boxes_num > 0
    colors = colormap('hsv');
    colors = colors(1:(floor(size(colors, 1)/valid_boxes_num)):end, :);
    colors = mat2cell(colors, ones(size(colors, 1), 1))';

    color_idx = 1;
    for i = 1:length(boxes)
        if isempty(boxes{i})
            continue;
        end

        for j = 1:size(boxes{i})
            box = boxes{i}(j, 1:4);
            score = boxes{i}(j, end);
            linewidth = 2 + min(max(score, 0), 1) * 2;
            rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors{color_idx});
            label = sprintf('%s : %.3f', legends{i}, score);
            text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
        end

        color_idx = color_idx + 1;
    end
    end
end

function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)

rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
end

