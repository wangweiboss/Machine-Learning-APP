clc,clear
load hald; %其中ingredients 为matlab自带  
X = zscore(ingredients); 
%% 方法1
[pc, score, latent,tsquare]=pca(X); % 不会给你去均值，拿来就算协方差 (X'*X)./n

%% 方法2
cov_ingredients =cov(X);% cov(X)去了均值，所以不等于(X'*X)./12，要用cov，最好将原始数据标准化;  %(X'*X)./12;
[V,D]=eig(cov_ingredients);    % D=latent,pc=v  

cov_ingredients1 =(X*X')./3;% cov(X');%(X*X')./3;用cov会有偏差
[V1,D1]=eig(cov_ingredients1);    
%V1和V的关系：V1(i)=X*V(i)/vv(倒着来,倒着来) 因为D和D1是从小到大排的，vv是从大到小排的，所以要倒着

%% 方法3
[ss,vv,dd] = svd(X);  %ss==V1,vv^2==D*12==D1*3,dd==V
%ss和dd的关系：ss(i)=X*dd(i)/vv(i,i) 要一列一列的除，否则会出现inf

注意：
1.cov有没有去均值
2.pca和princomp有没有去均值
3.ss和dd以及V和V1的转换用vv，但是dd、V、V1的归一化用latent
