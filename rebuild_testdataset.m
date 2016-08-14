function [P,T]=rebuild_testdataset(path,class,set,size1,base)

allsamples=[];
n=size(base,2);
k=0;
for i=1:class
    for j=set
        a=imread(strcat(path,num2str(i),'\',num2str(j),'.pgm'));
        b=a(1:size1);
        b=double(b);
        allsamples=[allsamples; b];
        k=k+1;
        T((i-1)*length(set)+k,i)=0.9;%人为打标签
    end
    k=0;
end

allcoor = allsamples * base;%测试集的人脸图像主分量200x71
[ P,PS]= mapminmax(allcoor);%归一化
% step2.测试集的重建
gx2(:,1:n)=P;
gx2(:,n+1:class+n)=T;
%测试样本顺序打乱
xd=gx2(randperm(numel(gx2)/(class+n)),:);
% xd=gx2;
gx=xd(:,1:n);d=xd(:,n+1:class+n);
P=gx';T=d';

end