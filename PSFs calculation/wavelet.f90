!***********************************************************************
        subroutine wavelet_forming(wavelet, lt, dt, f0, shift)

        implicit none
        !Dummy variables
         real::wavelet(lt)

        integer::lt
        real::f0,dt
        integer::shift
        !Local variables
        real,parameter::pi=3.1415926
        integer::it


        do it=1, lt
c         wavelet(it)=exp(-(pi*f0*((it-shift)*dt-1.0/f0))**2)*
c     +   (1-2*(pi*f0*((it-shift)*dt-1.0/f0))**2)
         wavelet(it)=exp(-(pi*f0*(it-shift)*dt)**2)*
     +   (1-2*(pi*f0*(it-shift)*dt)**2)

       enddo

       return
       end subroutine 

!***********************************************************************

          subroutine  covariance(wavelet, lt, wavelet_cor)

        implicit none
        integer::lt
        real::wavelet(lt)
        real::wavelet_cor(lt)

        integer::i, j


        wavelet_cor=0.0

        do i=1, lt
           do j=1, lt
            if(i+j-1 .gt. lt)goto 3333
             wavelet_cor(i)=wavelet_cor(i)+wavelet(j)*wavelet(j+i-1)
           enddo
3333      continue
          wavelet_cor(i)=wavelet_cor(i)/lt
         enddo

           end subroutine

!***********************************************************************












