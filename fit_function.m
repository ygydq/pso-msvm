%***************************************************************************
%º∆À„  ”¶∂»
%***************************************************************************
function fit = fit_function(present,X,Y,Xt,Yt)
%%
ker  = 'rbf';
tol  = 1e-20;
epsi = 1;
C=present(1);
par=present(2);
[Beta,NSV,Ktrain,i1] = msvr(X,Y,ker,C,epsi,par,tol);
Ktest = kernelmatrix(ker,Xt',X',par);
Ypredtest = Ktest*Beta;

fit=sum(sum((Yt-Ypredtest).^2));

