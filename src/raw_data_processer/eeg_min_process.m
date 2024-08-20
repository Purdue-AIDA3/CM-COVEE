function eeg_data_interp = eeg_min_process(all_eeg,destdirectory,secondary)
% This file requires EEGLAB and the Zapline extension to be installed !!!

zaplineConfig = struct('noisefreqs',[]);
zaplineConfig.coarseFreqDetectPowerDiff = 4;
zaplineConfig.noisefreqs = [];
zaplineConfig.chunkLength = 1e9;
zaplineConfig.adaptiveNremove = 1;
zaplineConfig.fixedNremove = 1;
zaplineConfig.plotResults = 0;
if secondary==0
    location = append(destdirectory,'Task 1 With Secondary Task');
else
    location = append(destdirectory,'Task 1 Without Secondary Task');
end

eeg_data = pop_loadset(all_eeg.name,location);
eeg_data.chanlocs.labels
eeg_data_clean = clean_artifacts(eeg_data,'FlatlineCriterion',5,'ChannelCriterion',0.85,'LineNoiseCriterion',4,'Highpass',[0.2500 0.7500],'BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
eeg_data_clean.chanlocs.labels
[eeg_data_filtered, com, b] = pop_eegfiltnew(eeg_data_clean, 'locutoff', 0.5,'hicutoff',40);
eeg_data_zapline = clean_data_with_zapline_plus_eeglab_wrapper(eeg_data_filtered,zaplineConfig); % specifying the config is optional
eeg_data_interp = eeg_interp(eeg_data_zapline, eeg_data.chanlocs, 'spherical');
eeg_data_interp.chanlocs.labels