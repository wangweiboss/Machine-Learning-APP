clc,clear
load hald; %����ingredients Ϊmatlab�Դ�  
X = zscore(ingredients); 
%% ����1
[pc, score, latent,tsquare]=pca(X); % �������ȥ��ֵ����������Э���� (X'*X)./n
%% ����2
cov_ingredients =cov(X);% cov(X)ȥ�˾�ֵ�����Բ�����(X'*X)./12��Ҫ��cov����ý�ԭʼ���ݱ�׼��;  %(X'*X)./12;
[V,D]=eig(cov_ingredients);    % D=latent,pc=v  

cov_ingredients1 =(X*X')./3;% cov(X');%(X*X')./3;��cov����ƫ��
[V1,D1]=eig(cov_ingredients1);    
%V1��V�Ĺ�ϵ��V1(i)=X*V(i)/vv(������,������) ��ΪD��D1�Ǵ�С�����ŵģ�vv�ǴӴ�С�ŵģ�����Ҫ����
%% ����3
[ss,vv,dd] = svd(X);  %ss==V1,vv^2==D*12==D1*3,dd==V
%ss��dd�Ĺ�ϵ��ss(i)=X*dd(i)/vv(i,i)