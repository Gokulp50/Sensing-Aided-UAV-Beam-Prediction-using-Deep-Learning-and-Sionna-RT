% DeepSense 6G - Testbed 1 Beam Patterns
%%

%%%%%%%%%%%%%%%% select the beams you want to plot %%%%%%%%%%%%%%%%
beam_set = 0:63;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_of_beam = length(beam_set);

codebook_pattern = zeros(num_of_beam, 211);


for ii = 1:num_of_beam
    data = load(['./codebook_beams/built_in_beam_pattern_',num2str(beam_set(ii)),'.mat']);
    beam_pattern = data.beam_pattern;
    beam_pattern(1) = beam_pattern(2); % the first value is always instable
    codebook_pattern(ii, :) = beam_pattern;
end

%%
measurement_offset_angle=4*pi/180;
angle_start = 0-measurement_offset_angle;
angle_end = pi-measurement_offset_angle;
num_of_angle = length(beam_pattern);

%%
fig_count = 1;
f = figure(fig_count);
for ii = 1:num_of_beam
    polarplot(linspace(angle_start, angle_end, num_of_angle), codebook_pattern(ii, :),  'linewidth', 1.2)
    fig_count = fig_count + 1;
    grid on
    hold on
end
