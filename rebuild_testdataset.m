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
        T((i-1)*length(set)+k,i)=0.9;%��Ϊ���ǩ
    end
    k=0;
end

allcoor = allsamples * base;%���Լ�������ͼ��������200x71
[ P,PS]= mapminmax(allcoor);%��һ��
% step2.���Լ����ؽ�
gx2(:,1:n)=P;
gx2(:,n+1:class+n)=T;
%��������˳�����
xd=gx2(randperm(numel(gx2)/(class+n)),:);
% xd=gx2;
gx=xd(:,1:n);d=xd(:,n+1:class+n);
P=gx';T=d';

end