      include  './wavelet.f90'

      program Hessian_calculate
      integer  nshot_final,nshot_step,nx,nz
      integer  mt,shift,kk,is
      integer  it,ishot_1,ixx,izz,kkk,itt

      real, allocatable :: time_s(:,:,:)
      real, allocatable :: time_r(:,:,:)
      real, allocatable :: time2d(:,:)
c      real, allocatable :: time(:,:) 

      real, allocatable :: wavelet(:)
      real, allocatable :: wavelet_dri(:)
      real, allocatable :: wavelet_cor(:)
      nx=1502
      nz=581
      nshot_final=751
      nshot_step=1
      shift=2000
      mt=4000
      allocate(wavelet(mt))
      allocate(wavelet_dri(mt))
      allocate(wavelet_cor(mt))
       
      allocate(time2d(nz,nx*nshot_final))
      allocate(time_s(nshot_final,nx,nz))
      allocate(time_r(nshot_final,nx,nz))
c      allocate(time(nshot_final*nx*nz,nz))
        

c    wavelet_cor colculate
      open(222,file='./data/wavelet_1ms.dat',
     +      access='direct',recl=1)     
       call wavelet_forming(wavelet,mt,0.001,100.0,shift)
        kk=1
        do is = 1,mt
           write(222,rec = kk)wavelet(is)
           kk = kk+1
        end do
        call  covariance(wavelet,mt,wavelet_cor)
        do is=1,mt
           wavelet_dri(is)=(wavelet_cor(is+2)+wavelet_cor(is)-2.0*
     +  wavelet_cor(is+1))*1000*1000
        end do
        do is=1,mt
           wavelet_dri(is)=(wavelet_dri(is+2)+wavelet_dri(is)-2.0*
     +  wavelet_dri(is+1))
        end do


        kk = 1
        open(7777,file='./wavelet_cor.dat',access = 'direct',recl=1)
c        call  covariance(wavelet_dri,mt,wavelet_cor)
          do is=1,mt
             write(7777,rec = kk)wavelet_dri(is)
             kk = kk + 1
          end do
        close(7777)
        close(222)


c      write(*,*)"times timer",nshot_final
c     traveltime transform
      open(333,file='./result/time_mar_nshot751.dat',
     +      access='direct',recl=nz)
      kkk=1
      do it=1,nx*nshot_final
          read(333,rec=kkk)time2d(:,it)
           kkk=kkk+1
      end do
      write(*,*)"times timer",nshot_final
      do it=0,nz*nx*nshot_final-1
          ishot_1=(it/(nz*nx))+1
          ixx=(mod(it,nz*nx)/nz)+1
          izz=mod(it,nz)+1
          time_s(ishot_1,ixx,izz)=time2d(izz,(ishot_1-1)*nx+ixx)
          time_r(ishot_1,ixx,izz)=time2d(izz,(ishot_1-1)*nx+ixx)
      end do      
      write(*,*)"times timer",nshot_final
      call Hessian_construct(nshot_final,nz,nx,time_s,time_r,
     +                              wavelet_dri,mt)
     
      end program

       SUBROUTINE Hessian_construct(nshot_final, nz,nx,time_s,time_r,
     +                              wavelet_dri,mt)
      
      integer  ishot,itrace,ix,ipsfx,jz,jpsfz,k,ii,jj,ttt1
      real  a_j,a_i,rtime_s,rtime_r,aa_1,ttt,amp_s,amp_r,dri
      dimension H(nx,nz),wavelet_dri(mt) 
      dimension time_s(nshot_final,nx,nz),time_r(nshot_final,nx,nz)
      k=1
      H=0.0
      write(*,*)"times timer",nshot_final
       do ishot = 1,nshot_final,5
        do itrace = 1,nshot_final,1
          if(abs(ishot-itrace) .gt. 300) cycle
c          do ix=60,nx-60,31
c            do ipsfx= ix-15,ix+15
c              do jz=50,nz-60,31
c                do jpsfz = jz-15,jz+15
          ix=750
          jz=290
          do ipsfx= ix-10,ix+10
            do jpsfz = jz-10,jz+10
c               if(time_s(ishot,ipsfx,jpsfz)==0.or.
c      +     time_r(itrace,ipsfx,jpsfz)==0.or.time_s(ishot,il,iz)==0
c      + .or.  time_r(itrace,il,jz)==0)cycle
               a_j=time_s(ishot,ipsfx,jpsfz) +time_r(itrace,ipsfx,jpsfz)
               a_i=time_s(ishot,ix,jz)+time_r(itrace,ix,jz)
               rtime_s=1./time_s(ishot,ipsfx,jpsfz)
               rtime_r=1./time_r(itrace,ipsfx,jpsfz)              
               amp_s=1./time_s(ishot,ix,jz)
               amp_r=1./time_r(itrace,ix,jz)

               ttt=abs((a_j-a_i))*1000
               ttt1=int(ttt)
               if(ttt1.lt.1)then
                  ttt1=1
               end if
c              aa_1=a_j*(rtime_s*rtime_s+rtime_r*rtime_r)
               aa_1=rtime_s*rtime_r*amp_s*amp_r
               H(ipsfx,jpsfz)=H(ipsfx,jpsfz)+aa_1*wavelet_dri(ttt1)

             end do
           end do
c          end do
c           end do
         end do
         write(*,*)"ishot is done",ishot
       end do

c read hessian on the desk 
       open(888,file='./result/psf_mar581x1502_100hz.dat',
     +       access='direct',recl=1)
         do ii=1,nx,1
          do jj=1,nz,1
           write(888,rec=k)H(ii,jj)
           k=k+1
          end do
         end do
      close(888)
      return
      end 

