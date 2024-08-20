clear
close all

rng(100);
words = readtable('.../all_nouns.txt');
words_nonan = words(~isnan(words.FAM),:);
words_nonan = words_nonan(~isnan(words_nonan.IMG),:);
words_nonan = words_nonan(~isnan(words_nonan.T_LFRQ),:);

words_nonan_matrix = [words_nonan.FAM words_nonan.IMG words_nonan.T_LFRQ];
words_nonan_matrix_norm = normalize(words_nonan_matrix);
idx = dbscan(words_nonan_matrix_norm,0.6,6);
words_selected = words_nonan(find(idx>0),:); % 2 words per video in task 1, 1 word every 10 seconds in task 2 => 210 words needed

task_1_idx = randperm(size(words_selected,1));
task_1_idx = task_1_idx(1:2*15);
task_2_idx = 1:size(words_selected,1);
task_2_idx = task_2_idx(~ismember(task_2_idx,task_1_idx));

task_1_words = words_selected(task_1_idx,:);
task_2_words = words_selected(task_2_idx,:);

task_1_words_matrix = [task_1_words.FAM task_1_words.IMG task_1_words.T_LFRQ];
task_2_words_matrix = [task_2_words.FAM task_2_words.IMG task_2_words.T_LFRQ];

task_1_words_matrix_norm = normalize(task_1_words_matrix,'range');
task_2_words_matrix_norm = normalize(task_2_words_matrix,'range');

task_1_words_dist_matrix = zeros(size(task_1_words,1),size(task_1_words,1));
task_2_words_dist_matrix = zeros(size(task_2_words,1),size(task_2_words,1));

for i=1:size(task_1_words,1)
    for j=1:size(task_1_words,1)
        task_1_words_dist_matrix(i,j) = sqrt(sum((task_1_words_matrix_norm(i,:)-task_1_words_matrix_norm(j,:)).^2));
    end
end

for i=1:size(task_2_words,1)
    for j=1:size(task_2_words,1)
        task_2_words_dist_matrix(i,j) = sqrt(sum((task_2_words_matrix_norm(i,:)-task_2_words_matrix_norm(j,:)).^2));
    end
end
%%
used = [];
group_task_1_words = [];
noun_idx = 1;
copy_matrix = task_1_words_dist_matrix;

for i=1:15
    while ~isempty(find(used==noun_idx))
        noun_idx = noun_idx + 1;
    end
    noun = task_1_words.WORD(noun_idx);
    temp = copy_matrix(noun_idx,:);
    noun_idx = noun_idx + 1;
    if ~isempty(used)
        temp(used) = Inf;
    end
    [temp_sorted,sort_idx] = sort(temp);
    group_task_1_words = [group_task_1_words; [noun task_1_words.WORD(sort_idx(2))]];
    used = [used;noun_idx-1 sort_idx(2)];
end
    
%%
used = [];
group_task_2_words = [];
noun_idx = 1;
copy_matrix = task_2_words_dist_matrix;

for i=1:2
    while ~isempty(find(used==noun_idx))
        noun_idx = noun_idx + 1;
    end
    noun = task_2_words.WORD(noun_idx);
    temp = copy_matrix(noun_idx,:);
    noun_idx = noun_idx + 1;
    used = [used;noun_idx-1];
    if ~isempty(used)
        temp(used) = Inf;
    end
    [temp_sorted,sort_idx] = sort(temp);
    temp_words = noun;
    if (temp_sorted(2)==0)
        disp("HEY!!!")
    end
    for j=2:90
        temp_words = [temp_words;task_2_words.WORD(sort_idx(j))];
        used = [used;sort_idx(j)];
    end
    group_task_2_words = [group_task_2_words temp_words];
    % used = [used;noun_idx-1 sort_idx(2)];
end


