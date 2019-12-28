function xr=recover(vec,xr,tol)

    n = size(vec,2);
    xrec = zeros(n,1);
    xrec(vec==0) = xr;
    xrec(abs(xrec)<=tol) = 0;
    xr = xrec;
    
end