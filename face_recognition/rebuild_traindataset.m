function [P,T,base]=rebuild_traindataset(path,class,set,size)

%% step1.PCA
allsamples=[];
k=0;
for i=1:class
    for j=set
        a=imread(strcat(path,num2str(i),'\',num2str(j),'.pgm'));
        b=a(1:size);
        b=double(b);
        allsamples=[allsamples; b];
        k=k+1;
        T((i-1)*length(set)+k,i)=0.9;%人为打标签
    end
    k=0;
end

samplemean=mean(allsamples); %求出每行的平均值
for i=1:class*length(set)
    xmean(i,:)=allsamples(i,:)-samplemean; % 平均脸
end
% 获取特征值及特征向量
sigma=xmean*xmean'; %形成偏差矩阵
[v d]=eig(sigma);% sigma*V = V*D.
d1=diag(d);
% 按特征值大小以降序排列
dsort = flipud(d1);
vsort = fliplr(v);
%以下选择 90%的能量
dsum = sum(dsort);
dsum_extract = 0;
p = 0;
while( dsum_extract/dsum < 0.9)%选取90%的主成分
    p = p + 1;
    dsum_extract = sum(dsort(1:p));
    ddd(p)=dsum_extract/dsum;
end

%% step2.重构训练集training set
% 计算特征脸形成的坐标系
base = xmean' * vsort(:,1:p) * diag(dsort(1:p).^(-1/2));%正交归一特征向量

allcoor = allsamples * base;%训练集的人脸图像主分量200x71
[P PS]= mapminmax(allcoor);%归一化
% step2.训练集的重建
gx2(:,1:p)=P;
gx2(:,p+1:p+class)=T;
%训练样本顺序打乱
xd=gx2(randperm(numel(gx2)/(p+class)),:);
% xd=gx2;
gx=xd(:,1:p);d=xd(:,p+1:p+class);
P=gx';T=d';

end
