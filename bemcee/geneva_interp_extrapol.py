def geneva_interp(Mstar, oblat, t, Zstr='014', tar=None, silent=True):
    '''
    Interpolates Geneva stellar models.

    Usage:
    Rpole, logL, age = geneva_interp(Mstar, oblat, t, tar=None, silent=True)

    where t is given in tMS, and tar is the open tar file. The chosen
    metallicity is according to the input tar file. If tar=None, the
    code will take Zstr='014' by default.
    '''
    # oblat to Omega/Omega_c
    # w = oblat2w(oblat)

    # grid
    if Mstar <= 20.:
        Mlist = _np.array([1.7, 2., 2.5, 3., 4., 5., 7., 9., 12., 15., 20.])
    else:
        Mlist = _np.array([20., 25., 32., 40., 60., 85., 120.])

    # read tar file
    if tar is None:
        dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
        fmod = 'Z{:}.tar.gz'.format(Zstr)
        tar = _tarfile.open(dir0 + fmod, 'r:gz')
    else:
        Zstr = tar.getnames()[0][7:10]

    # interpolation
    
    # creation of lists for polar radius and log, for extrapolation fit
    # and the ttms list used in the linear fit originally
    
    ttms = [0, 0.40, 0.65, 0.85, 1.00]
    L_log = []
    Rp = []
    
    
    # for ages inside the original grid, nothing happens
    if (t < 1.001) * (t >= 0.):    
		if (Mstar >= Mlist.min()) * (Mstar <= Mlist.max()):
			if (Mstar == Mlist).any():
				Rpole, logL, age = geneva_closest(Mstar, oblat, t, tar=tar, 
					Zstr=Zstr, silent=silent)								
        else:
            # nearest value at left
            Mleft = Mlist[Mlist < Mstar]
            Mleft = Mleft[_np.abs(Mleft - Mstar).argmin()]
            iMleft = _np.where(Mlist == Mleft)[0][0]
            Rpolel, logLl, agel = geneva_closest(Mlist[iMleft], oblat, t, 
                tar=tar, Zstr=Zstr, silent=silent)
            # nearest value at right
            Mright = Mlist[Mlist > Mstar]
            Mright = Mright[_np.abs(Mright - Mstar).argmin()]
            iMright = _np.where(Mlist == Mright)[0][0]
            Rpoler, logLr, ager = geneva_closest(Mlist[iMright], oblat, t, 
                tar=tar, Zstr=Zstr, silent=silent)
            # interpolate between masses
            weight = _np.array([Mright-Mstar, Mstar-Mleft]) / (Mright-Mleft)
            Rpole = weight.dot(_np.array([Rpolel, Rpoler]))
            logL = weight.dot(_np.array([logLl, logLr]))
            age = weight.dot(_np.array([agel, ager]))
		else:
			if not silent:
				print('[geneva_interp] Warning: Mstar out of available range, '
					'taking the closest value.')
			Rpole, logL, age = geneva_closest(Mstar, oblat, t, tar=tar, Zstr=Zstr, 
				silent=silent)

		return Rpole, logL, age
		
	if (t > 1.001):		
		for time in ttms:
			if (Mstar >= Mlist.min()) * (Mstar <= Mlist.max()):
				if (Mstar == Mlist).any():
					Rpole, logL, age = geneva_closest(Mstar, oblat, t, tar=tar, 
						Zstr=Zstr, silent=silent)
						
					Rp.append(Rpole)
					L_log.append(logL)								
			else:
				# nearest value at left
				Mleft = Mlist[Mlist < Mstar]
				Mleft = Mleft[_np.abs(Mleft - Mstar).argmin()]
				iMleft = _np.where(Mlist == Mleft)[0][0]
				Rpolel, logLl, agel = geneva_closest(Mlist[iMleft], oblat, t, 
					tar=tar, Zstr=Zstr, silent=silent)
				# nearest value at right
				Mright = Mlist[Mlist > Mstar]
				Mright = Mright[_np.abs(Mright - Mstar).argmin()]
				iMright = _np.where(Mlist == Mright)[0][0]
				Rpoler, logLr, ager = geneva_closest(Mlist[iMright], oblat, t, 
					tar=tar, Zstr=Zstr, silent=silent)
				# interpolate between masses
				weight = _np.array([Mright-Mstar, Mstar-Mleft]) / (Mright-Mleft)
				Rpole = weight.dot(_np.array([Rpolel, Rpoler]))
				logL = weight.dot(_np.array([logLl, logLr]))
				age = weight.dot(_np.array([agel, ager]))
				
				Rp.append(Rpole)
				L_log.append(logL)	
				
			else:
				if not silent:
					print('[geneva_interp] Warning: Mstar out of available range, '
					'taking the closest value.')
				Rpole, logL, age = geneva_closest(Mstar, oblat, t, tar=tar, Zstr=Zstr, 
					silent=silent)
					
				Rp.append(Rpole)
				L_log.append(logL)	
				
		coeffs = _np.polyfit(_np.log10(ttms[-4:]), _np.log10(Rp[-4:]), deg=1)
		poly = _np.poly1d(coeffs)
        
		coeffs2 = _np.polyfit(ttms[-4:], L_log[-4:], deg=1)
		poly2 = _np.poly1d(coeffs2)
		
		Rpole = 10**(poly(np.log10(t)))
		logL = (poly2(t))
		
		
		# in this case, no age. no physical meaning!
		return Rpole, logL
		
