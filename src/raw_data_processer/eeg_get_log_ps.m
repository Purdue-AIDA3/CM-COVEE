function all_bands = eeg_get_log_ps(eeg_data)
data = eeg_data.data;
data = data(:,1:(floor(size(data,2)/eeg_data.srate)*eeg_data.srate));
window_data = {};

for i=1:size(data,2)/eeg_data.srate
    window_data{i} = data(:,((i-1)*eeg_data.srate+1):i*eeg_data.srate);
end

all_bands = [];
for j=1:length(window_data)
    [eegspecdB,freqs,compeegspecdB,resvar,specstd] = spectopo(window_data{j}, 0, eeg_data.srate,'plot', 'off');

    all_bands(j,:,:) = eegspecdB;
end