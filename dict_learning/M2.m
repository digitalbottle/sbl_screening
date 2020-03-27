% M2 function
function result = M2(t1,t2,t3)
    global n1;
    global n2;
    global r;
    global phi1;
    global phi2;    
    tau = n1'*n2;
    if ( t1< -phi1*t3 && t2< -phi2*t3 )
        result = r*t3;
    else
        if ( t2 >= -phi2*t3 && (t1-tau*t2)/sqrt(t3^2-t2^2)<(-phi1+tau*phi2)/sqrt(1-phi2^2) )
            result = -r*t2*phi2+r*sqrt(t3^2-t2^2)*sqrt(1-phi2^2);
        else
            if ( t1 >= -phi1*t3 && (t2-tau*t1)/sqrt(t3^2-t1^2)<(-phi2+tau*phi1)/sqrt(1-phi1^2) )
                result = -r*t1*phi1+r*sqrt(t3^2-t1^2)*sqrt(1-phi1^2);
            else
                result = -r/(1-tau^2)*( (phi1-tau*phi2)*t1 + (phi2-tau*phi1)*t2 ) + ...
                    r/(1-tau^2)*sqrt( 1-tau^2+2*tau*phi1*phi2-phi1^2-phi2^2 )*sqrt(...
                    (1-tau^2)*t3^2+2*tau*t1*t2-t1^2-t2^2 );
            end
        end
    end
end