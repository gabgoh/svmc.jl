function plotdb(A, y, x, ρ)

	figure()

	plot( A[ y .== -1 , 1] , A[ y .== -1 , 1],"r." )
	plot( A[ y .==  1 , 1] , A[ y .==  1 , 1],"b." )

	decbound(y)  = (ρ - x[1]*y)/x[2]
	decbound2(y) = (-1 + ρ - x[1]*y)/x[2]

	ax = axis();
	plot( [ ax[1]; ax[2] ], [ decbound(ax[1])  ; decbound(ax[2])  ] , "k"   )
	plot( [ ax[1]; ax[2] ], [ decbound2(ax[1]) ; decbound2(ax[2]) ] , "k--" )

	axis(ax);

end