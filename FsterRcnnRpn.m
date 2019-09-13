function FsterRcnnRpn(para, opts, proposal_detection_model, rpn_net, mode, average)   
% average = cell2mat(average);
tic
    Files = para.Files; ReturnBox_all = [];Boxes_all = [];
    for img_index = 1:para.LengthFiles-1    
            ImageName = Files(img_index).name;
            curFramePicture = imresize(imread(['.\sequence\' mode '\Imgs\' ImageName]),[300,300]);
            curLowPicture = imread([para.LowlevelSaliencyPath ImageName(1:end-4) '.jpg']);
            curlowRect = get_Rect_ori(curLowPicture);
            curlowRect = curlowRect(:,[3,1,4,2]);
%             if img_index == 1
%                 saveReturnBox = curlowRect;
%             end
                       
            if ismember(img_index, average)
                curlowRect = curlowRect;
%                 showboxesc(curLowPicture,curlowRect,'r','-');
            else
                im= uint8(curFramePicture); 
                if opts.use_gpu
                    im = gpuArray(im);
                end
                [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
                aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
                o = boxoverlap(curlowRect, round(aboxes(:,1:4)));
                [score,index]=max(o);
                curlowRect = floor(aboxes(index,1:4));
%                 showboxesc(curLowPicture,curlowRect,'r','-');
            end         
%             saveReturnBox = ReturnBox;
            ReturnBox_all = [ReturnBox_all;curlowRect];
%             showboxesc(curFramePicture,curlowRect,'r','-');
%             showboxesc(curLowPicture,curlowRect,'r','-')
%                ReturnBox = ReturnBox(:,[2 4 1 3]);%raw and colum%
            %将得到的rect进行保存
%             save([para.RPNBoxPath ImageName(1:end-4) '_box.mat'],'ReturnBox');
    end

    Width = ReturnBox_all(:,3)-ReturnBox_all(:,1); Height = ReturnBox_all(:,4)-ReturnBox_all(:,2);
    Middle = [ReturnBox_all(:,1)+Width/2, ReturnBox_all(:,2)+Height/2];  
    MidWidHei = [Middle, Width, Height];
        All_N = size(MidWidHei,1);
    for picindex = 1:All_N
        ImageName = Files(picindex).name;
        curFramePicture = imresize(imread(['.\sequence\' mode '\Imgs\' ImageName]),[300,300]);
%            if ismember(picindex, average)
%                 ReturnBox = MidWidHei(picindex, :);
% %                 showboxesc(curFramePicture,ReturnBox,'r','-');
%                 ReturnBox = ReturnBox(:,[2 4 1 3]);%raw and colum%
%                 save([para.RPNBoxPath ImageName(1:end-4) '_box.mat'],'ReturnBox');
%                 continue;
%            end
%         ReturnBox=ReturnBox_all(picindex, :);
        if picindex>2
            ReturnBox_pre1=MidWidHei(picindex-1, :);
            ReturnBox_pre2=MidWidHei(picindex-2, :);
        else
            ReturnBox_pre1=MidWidHei(picindex, :);
            ReturnBox_pre2=MidWidHei(picindex, :);
        end
        if picindex<All_N-1
            ReturnBox_lat1=MidWidHei(picindex+1, :);
            ReturnBox_lat2=MidWidHei(picindex+2, :);
        else
            ReturnBox_lat1=MidWidHei(picindex, :);
            ReturnBox_lat2=MidWidHei(picindex, :);
        end
           Boxes = mean([ReturnBox_pre1;ReturnBox_pre2;ReturnBox_lat1;ReturnBox_lat2]);
           Boxes = [Boxes(1,1)-Boxes(1,3)/2,Boxes(1,2)-Boxes(1,4)/2,Boxes(1,1)+Boxes(1,3)/2,Boxes(1,2)+Boxes(1,4)/2];

           Boxes_all = [Boxes_all; Boxes];
%            showboxesc(curFramePicture,Boxes,'r','-');
    end
    toc
        Boxes_all = floor(Boxes_all(:,[2 4 1 3]));%raw and colum%
        save([para.RPNBoxPath 'Boxes_all.mat'],'Boxes_all');
end
