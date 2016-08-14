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
        T((i-1)*length(set)+k,i)=0.9;%��Ϊ���ǩ
    end
    k=0;
end

samplemean=mean(allsamples); %���ÿ�е�ƽ��ֵ
for i=1:class*length(set)
    xmean(i,:)=allsamples(i,:)-samplemean; % ƽ����
end
% ��ȡ����ֵ����������
sigma=xmean*xmean'; %�γ�ƫ�����
[v d]=eig(sigma);% sigma*V = V*D.
d1=diag(d);
% ������ֵ��С�Խ�������
dsort = flipud(d1);
vsort = fliplr(v);
%����ѡ�� 90%������
dsum = sum(dsort);
dsum_extract = 0;
p = 0;
while( dsum_extract/dsum < 0.9)%ѡȡ90%�����ɷ�
    p = p + 1;
    dsum_extract = sum(dsort(1:p));
    ddd(p)=dsum_extract/dsum;
end

%% step2.�ع�ѵ����training set
% �����������γɵ�����ϵ
base = xmean' * vsort(:,1:p) * diag(dsort(1:p).^(-1/2));%������һ��������

allcoor = allsamples * base;%ѵ����������ͼ��������200x71
[P PS]= mapminmax(allcoor);%��һ��
% step2.ѵ�������ؽ�
gx2(:,1:p)=P;
gx2(:,p+1:p+class)=T;
%ѵ������˳�����
xd=gx2(randperm(numel(gx2)/(p+class)),:);
% xd=gx2;
gx=xd(:,1:p);d=xd(:,p+1:p+class);
P=gx';T=d';

end