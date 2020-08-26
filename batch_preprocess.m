function batch_preprocess(path)
num_antennas=4;
data_shape=[512   128   400     1];


sub_folder_list=dir(path);%'/data/datasets/external/Fast');

dat_all=[]; % data matrix
des_all=[]; % design matrix

for idx=1:size(sub_folder_list,1)
 file_path=[sub_folder_list(idx).folder,'/',sub_folder_list(idx).name,'/adc_Data.mat'];

 if ~exist(file_path)
     continue
 end

 dat_tmp=load(file_path);


 dat_tmp=table2array(dat_tmp.adc_DataTable);
    for idx2=1:num_antennas;

        feat_tmp=feature_process(dat_tmp(:,idx2),data_shape,true);
        dat_all=cat(3,dat_all,feat_tmp);
        class_tmp=split(sub_folder_list(idx).folder,'/');
        class_tmp=class_tmp(end);
        des_tmp=struct('person',sub_folder_list(idx).name,'Class',class_tmp,'Antenna',idx2);
        des_all=cat(1,des_all,des_tmp);

    end
end
save([path,'/pre_processed.mat'],'dat_all')
writetable(struct2table(des_all),[path,'/pre_processed.csv'])
end
