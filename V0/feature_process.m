function feature_set=feature_process(raw_data,data_shape,decluter)
    window_size=256;
    overlap_size=192;
    fft_size=256;
    decluter_band_width=3;
    
    dat=reshape(raw_data,data_shape(1),data_shape(2),data_shape(3),data_shape(4));
    dat=sum(dat,1);% collapse range
    dat=dat(:);
    feature_set=spectrogram(dat,window_size,overlap_size,fft_size,'centered');
    if decluter
       feature_set((int32(fft_size-decluter_band_width)/2):int32((fft_size+decluter_band_width)/2),:)=0 ;
    end

end
