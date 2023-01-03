


% generate 100 random face sequences 
rng('default');
M = 100;
xstd = 1;
ystd = 1;
fvals = linspace(400,1500,12);
w = 640;
h = 480;

load('Model_Shape.mat');

%% create face sequence
for f = fvals
    for i = 1:10
        % get a sequence 
        while true
            [xw,alphas] = generateRandomFace(mu_shape,sigma,shape_eigenvec);
            lm = xw(:,keypoints)/ 1000;
            lm = lm - mean(lm,2);
            lm(3,:) = lm(3,:) * -1;
            
            % set principal point somewhere in the center of the image
            px = (w/2) + randn * 10;
            py = (h/2) + randn * 10;
            
            
            sequence = generateFaceSequence2(M,lm,xstd,ystd,f,px,py);
   
%             sequence = generateRandomFaceSequence(M,N,xstd,ystd,f);
            xmin = min(min(sequence.x_img(:,:,1),[],2));
            xmax = max(max(sequence.x_img(:,:,1),[],2));
            ymin = min(min(sequence.x_img(:,:,2),[],2));
            ymax = max(max(sequence.x_img(:,:,2),[],2));

            if xmin >= 0 && ymin >= 0 && xmax <= w && ymax <= h
                break;
            end
        end
        
        outdir = sprintf('../../data/synthetic_principal/sequencef%04d',f);
        outfile = sprintf('sequence%03d.mat',i);
        fnameout = [outdir '/' outfile];
        if ~exist(outdir,'dir')
            mkdir(outdir);
        end
        disp(fnameout);
        save(fnameout,'sequence');
    end
end


