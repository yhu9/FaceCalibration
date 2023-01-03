
% % % INPUT: 
% M         scalar
% xstd      scalar
% ystd      scalar
% f         scalar
%
% % % OUTPUT:
% sequence      struct
% x_w         (68x3)
% x_cam       (Mx68x3)
% x_i       (Mx68x2)
% R           (Mx3x3)
% T           (Mx3)
% 
function sequence = generateFaceSequence2(M,x_w,xstd,ystd,f,px,py)
   
    % define output
    N = size(x_w,2);
    x_cam = zeros(M,N,3);
    x_i = zeros(M,N,2);
    x_i_gt = zeros(M,N,2);
    Quaternion = zeros(M,4);
    R = zeros(M,3,3);
    T = zeros(M,3);

    % define bounds on extrinsics as constants change if you feel like
    minz = 400; maxz = 4000;
    tz = rand*(maxz - minz) + minz;
    minz = max(tz - 2000,minz);
    maxz = min(tz + 2000,maxz);
    max_rx = 20;
    max_ry = 30;
    max_rz = 20;
    max_delta_dist = 5;
    max_delta_rot = 0.1*pi/180;
    
    % define 
    K = [f,0,px; 0,f,py; 0,0,1];
    
    % random rotation within bounds. Initialize frame 0
    [~,~,~,q_init] = generateRandomRot(max_rx,max_ry,max_rz);
    [~,~,~,q_final] = generateRandomRot(max_rx,max_ry,max_rz);
    
    q1 = quaternion(q_init);
    q2 = quaternion(q_final);
    
    % random translation within bounds. initialize frame 0
    t_init = generateRandomTranslation(K,minz,maxz);
    t_final = generateRandomTranslation(K,minz,maxz);
    
    % spherical linear interpolation of the angles
    Q = slerp(q1,q2,linspace(0,1,100));    
    V = [linspace(t_init(1),t_final(1),M); linspace(t_init(2),t_final(2),M);...
        linspace(t_init(3),t_final(3),M)];
    
    % create rest of frames according to rotation and translation
    for i = 1:M
        % get rotation
        q = Q(i);
        r = rotmat(q,'point');
        thetas = rotvec(q);
        rx = thetas(1);
        ry = thetas(2);
        rz = thetas(3);
        Quaternion(i,:) = compact(q);
        R(i,:,:) = r;
        
        % get translation
        t = V(:,i);
        T(i,:) = t';

        % project onto 2d
        xc = (r*x_w + t);
        x_cam(i,:,:) = xc';
        proj = K * xc;
        proj = proj ./ proj(3,:);
        x2d = proj';
        x2d = x2d(:,1:2);
        x_i_gt(i,:,:) = x2d;
        
        % add 2d noise
        xnoise = randn(68,1)*xstd;
        ynoise = randn(68,1)*ystd;
        x_i(i,:,:) = x2d + [xnoise,ynoise];
    end
    
    c = squeeze(mean(x_cam,2));
    d = vecnorm(c');
    minvals = squeeze(min(x_i,[],2));
    maxvals = squeeze(max(x_i,[],2));
    diff = abs(minvals - maxvals);
    a = diff(:,1).*diff(:,2);
    a = sqrt(a);
    a = a';
    
    sequence.x_w = double(x_w)';
    sequence.x_cam = x_cam;
    sequence.x_img_true = x_i_gt;
    sequence.x_img = x_i;
    sequence.K = K;
    sequence.Quaternion = Quaternion;
    sequence.R = R;
    sequence.T = T;
    sequence.f = f;
    sequence.xstd = xstd;
    sequence.ystd = ystd;
    sequence.max_rx = rx;
    sequence.max_ry = ry;
    sequence.max_rz = rz;
    sequence.d = d;
    sequence.a = a;
    sequence.max_delta_dist = max_delta_dist;
    sequence.max_delta_rot = max_delta_rot;
end


function [rx,ry,rz,q] = generateRandomRot(max_rx,max_ry,max_rz)
% random rotation within bounds. Initialize frame 0
    while true
        ax = max_rx * pi / 180;
        ay = max_ry * pi / 180;
        az = max_rz * pi / 180;
        rx = rand*2*ax - ax;
        ry = rand*2*ay - ay;
        rz = rand*2*az - az;
        q = eul2quat([rz,ry,rx]);

        x_degree = rx * 180 / pi;
        y_degree = ry*180/pi;
        z_degree = rz*180/pi;
        if(abs(x_degree) <= max_rx && abs(y_degree) <= max_ry && abs(z_degree) <= max_rz)
            break
        end
    end
end

function t = generateRandomTranslation(K,minz,maxz)
    w = K(1,3)*2;
    h = K(2,3)*2;
    vx = inv(K)*[w;w/2;1];
    vy = inv(K)*[h/2;h;1];
    vz = [0;0;1];
    thetax = atan2(norm(cross(vz,vy)),dot(vz,vy));
    thetay = atan2(norm(cross(vz,vx)),dot(vz,vx));
    
    tz = rand*(maxz-minz) + minz;
    
    maxx = tz * tan(thetax);
    maxy = tz * tan(thetay);
    tx = rand*maxx*2 - maxx;
    ty = rand*maxy*2 - maxy;
    t = [tx,ty,tz];
end
