% ---------------------------------------------------------------------------------------------------
% KERNELMATRIX
%
% K = kernelmatrix(ker,X,X2,parameter);
%
% Builds a kernel from training and test data matrices. 
%
% Inputs: 
%	ker: {'lin' 'poly' 'rbf'}
%	X:	data matrix with training samples in rows and features in columns
%	X2:	data matrix with test samples in rows and features in columns
%	parameter: 
%       width of the RBF kernel
%       bias in the linear and polinomial kernel 
%       degree in the polynomial kernel
%
% Output:
%	K: kernel matrix
%
% Gustavo Camps-Valls, 2008(c)
% gcamps@uv.es, http://www.uv.es/gcamps
%
% ---------------------------------------------------------------------------------------------------

function K = kernelmatrix(ker,X,X2,parameter);

if  strcmp(ker,'lin')
	if ~isempty(X2)
	  K = X' * X2/(norm(X'*X2)) + parameter;
	else
	  K = X' * X/(norm(X'*X2))  + parameter;
	end;
	
elseif  strcmp(ker,'poly')
	if ~isempty(X2)
	  K = (X'* X2/(norm(X'*X2)) + 1).^parameter;
	else
	  K = (X'*X/(norm(X'*X2))  + 1).^parameter;
	end;

elseif strcmp(ker,'rbf')
	
	n1sq = sum(X.^2);
	n1   = size(X,2);
	
    if n1==1 % just one feature
        N1=size(X,1);
        N2=size(X2,1);
        D = zeros(N1,N2);
        for i=1:N1
            D(i,:) = (X2-ones(N2,1)*X(i,:))'.*(X2-ones(N2,1)*X(i,:))';
        end
    else
        if isempty(X2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq - 2*X'*X2;
        end;
    end
    K = exp(-D.^2/(2*parameter^2));
end;
