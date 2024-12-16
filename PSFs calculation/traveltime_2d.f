*==================================================================*
*       REMARKS:                                                   *
*         WRITEN BY : LIUSY                                        *
*                 FOR NANJING SOFTWARE DEVELOPMENT                 *
*                 12, 2009                                         *      
*       SUMMARY:                                                   *
*         THIS PROGRAM DO A 2D KIRCHHOFF PRESTACK DEPTH            *
*       MIGRATION WITH COMMON-OFFSET GATHER.                       *
*         ONE OF THE IMPORTANT PART OF THIS PROGRAM  IS TRAVEL     *
*       CALCULATION. THIS WAS FINISHED BY  HERB. Wang .            *
*==================================================================*
*===============================================================*
*                                                               *
*             TRAVLE TIME CALCULATION                           *
*                     2 - D                                     * 
*        AUTHOR  :  HERB. Wang                                  *
*        DATE    :  97,9                                        *
*        Modified By: Herb. Wang for iCluster                   *  
*        Date    :  08,3                                        *   
*        REFERENCE :                                            *
*               Schneider, W. A., et al , 1992, A dynaic        *
*        programming approach to first arrival traveltime       *
*        computation in media with arbitrary distributed        *
*        velocities. Geophysics, Vol.57, NO.1, P39-50.          * 
*                                                               *
*     NOTE: This program can calculate the traveltime wherever  *
*     the shot point stay.                                      * 
*===============================================================*

c      include './wavelet.f90'
     
      PROGRAM TRAVELTIME_FOR_2D_MODEL

*=====Global variable================================================ 

      PARAMETER(Pai=3.14159265359)
      PARAMETER(Lmax=100000000, Lbyte=1,NCDP_MAX=3400)
      CHARACTER(LEN=256) FN1, FN2, FN3, FN4, FN5, FN6

*=====Travel time variable(variable for travel time caculated.)======

      INTEGER NVCDP_START, NVCDP_FINAL
      INTEGER NTP, NVX, NVZ 

      REAL    VX_START, VZ_START

      REAL    DTSX, SX_START, SZ_START, SX_LEFT, SX_RIGHT
      REAL    DVX, DVZ, DEPTH_image, DZ

      INTEGER NTSX
      INTEGER L1, L2, L3, L4, L5, L6, L7
      INTEGER LNEED,LNEED2

      INTEGER,ALLOCATABLE :: BUFF(:)
      REAL,ALLOCATABLE :: TEMP1(:,:)
      REAL,ALLOCATABLE :: TEMP2(:,:)


*========================================================================================
*
*     FN1 : THE VELOCITY FILENAME
*     FN2 : THE TRAVEL TIME FILENAME
*     NCMP_first : The starting CMP number for Kirchhof PSDM
*     RCMP_first_COORD :The starting CMP coodinate      
*     NCMP_total : The total CMP number for Kirchhof PSDM
*     DCMP : The CMP interval
*     Offset_max : The maximum Offset
*     Aperture_mig_max : The maximum migration aperture      
*     NTP : TRAVLETIME CALCULATION POINT NUMBER
*     DS_X : TRAVLETIME CALCULATION POINT INTERVAL
*     SX_START : THE FIRST TRAVLETIME CALCULATION POINT POSITION
*     SX_LEFT  : THE LEFT SIDE DISTANCE TO RAVLETIME CALCULATION POINT
*     SX_RIGHT : THE RIGHT SIDE DISTANCE TO RAVLETIME CALCULATION POINT
*     VX_START, VZ_START : THE START COORDINTATE OF THE VELOCITY FIELD
*     NVX, NVZ : THE VELOCITY FIELD DIMENSION(NVX, NVY, NVZ)
*     DVX, DVZ : THE VELOCITY FIELD SAMPLE INTERVAL
*     DEPTH : THE TRAVELTIME CALCULATION DEPTH
*     DZ : THE TRAVELTIME CALCULATION DEPTH STEPLENGTH
*      
*========================================================================

      SX_START = 0.0
      SZ_START = 0.0
       write(*,*) 'SX_START, SZ_START'
       write(*,*)  SX_START, SZ_START       
c       write(*,*)' DVX, DVZ=', DVX, DVZ
c       write(*,*)' DS_X=', DS_X
c      pause
      NTP = 100
      
      SX_LEFT = 0.0 
      SX_RIGHT= 7505.0
      NZ = 581
      DVX =5.0
      DVZ =5.0 
      DX = 5.0
      DZ = 5.0
      DTSX = 10.0
  
      NXS_left=SX_LEFT/DVX+0.5
      NXS_right=SX_RIGHT/DVX+0.5      
      NX=NXS_left+NXS_right+1    
  
      Krec_save=1      

      L1=1
      L2=L1+(NX+1)
      L3=L2+(NX+1)
      L4=L3+NX
      L5=L4+NX
      L6=L5+(NX+1)*(NZ+1)
      LNEED = L6 +NX*NZ
      ALLOCATE (BUFF(LNEED))

      WRITE(*,*) '============================================='
      WRITE(*,*) '    2D TRAVEL TIME CALCULATING BEGIN         '
      WRITE(*,*) '============================================='

      OPEN(12, FILE='./result/time_mar_nshot751.dat', ACCESS='DIRECT', 
     +         RECL=Lbyte*NZ)


      DO 666 IS=1, 1
c      DO 666 IS=1, NTP

c       SX_COORD=SX_START+(IS-1)*DTSX 
       SX_COORD=SX_START
       SZ_COORD=SZ_START

       NS_X=SX_COORD/DVX+0.5
       NS_X2=NS_X
       IF(NS_X2.lt.1)NS_X2=1
       if(NS_X2.gt.NCDP_MAX)NS_X2=NCDP_MAX
       NS_Z = 1
       
       write(*,*)'IS=',IS
       write(*,*) 'SX_COORD, SZ_COORD=' 
       write(*,*) SX_COORD, SZ_COORD   
       write(*,*) 'NS_X, NS_Z=', NS_X, NS_Z
       write(*,*) 'NX, NZ=', NX, NZ

       CALL TRAVELTIME_2D_MAIN(BUFF(L1), BUFF(L2), BUFF(L3), BUFF(L4), 
     +      BUFF(L5), BUFF(L6), 
     +     
     +      NVX, NX, DX, NVZ, NZ, DZ, NS_X, NS_Z,
     +      NXS_left, NXS_right, Krec_save)

c       pause 888
666   CONTINUE
      DEALLOCATE(BUFF)
      WRITE(*,*) '============================================='
      WRITE(*,*) '    2D TRAVEL TIME CALCULATING FINISHED      '
      WRITE(*,*) '============================================='
      
      CLOSE(12)  
      END 

*===========================================================================

      SUBROUTINE TRAVELTIME_2D_MAIN(SS1, SS2, TT1, TT2, SLOWNESS, TIME,
     +
     +      NVX, NX, DX, NVZ, NZ, DZ, NS_X, NS_Z, 
     +      NXS_left, NXS_right, Krec_save)
 
      INTEGER NX, NZ, NS_X, NS_Z, NXS_left, NXS_right, ix, k, NSHOT
      INTEGER SHOT_STEP,lt,shift
      REAL    DX, DZ
c      real, allocatable :: wavelet_cor(:)
c      real, allocatable :: wavelet(:)
c      real, allocatable :: wavelet_dri(:)
c      real, allocatable :: time2d(:)
 
      DIMENSION SS1(0:NX), SS2(0:NX), TT1(NX), TT2(NX)
      DIMENSION SLOWNESS(0:NX, 0:NZ), TIME(NX, NZ)
      
c      real, allocatable :: time_s(:,:,:)
c      real, allocatable :: time_r(:,:,:)
c       lt = 500

c       shift = 250
       NSHOT = 751
       SHOT_STEP = 1
c       allocate(wavelet(lt))
c       allocate(wavelet_dri(lt))
c       allocate(wavelet_cor(lt))
c       allocate(time2d(NZ*NX*NSHOT))
c       allocate(time_s(NSHOT,NX,NZ))
c       allocate(time_r(NSHOT,NX,NZ))

        

*===== Zeroing the working buffer

      CALL ZERO_BUF(NX, NZ, SS1, SS2, TT1, TT2, TIME, SLOWNESS)

*===== Read the current-shot velocity into the working buffer SLOWNESS

      CALL READ_CURRENT_SHOT_VELO(NS_X, NXS_left, NXS_right, 
     +                  NVX, NX, NVZ, NZ, SLOWNESS)
      DO iShot = 1, NSHOT, SHOT_STEP
        NS_X = iShot*2
c      DS_Z = 0 
c      DS_X = 0
        DS_X = NS_X*DX-5
c      DS_Z = NS_Z*DZ     
        NS_Z = 1
        DS_Z = 0     
        write(*,*) 'nx=', nx, 'dx=', dx
        write(*,*) 'nz=', nz, 'dz=', dz
        write(*,*) 'DS_X=', DS_X, 'DS_Z=', DS_Z, 'NS_X=',NS_X
c     pause
         CALL TRAVELTIME_2D(SS1, SS2, TT1, TT2, SLOWNESS, TIME, NX,
     +           DX, NZ, DZ, NS_X, NS_Z, DS_X, DS_Z)
     
c      DO IZ=1, NZ
c       DO IX=1, NX
c        TIME(IX, IZ)= SLOWNESS(IX, IZ)*
c     +                              SQRT((IX*DX-DS_X)*(IX*DX-DS_X)
c     +                                  +(IZ*DZ-DS_Z)*(IZ*DZ-DS_Z))
c       END DO
c      END DO      
      
        CALL WRITE_DISK(NX, NZ, Krec_save, TIME)
      enddo 
*=================== Define wavelet and covariance ======

c       open(222,file='wavelet.dat',access='direct',recl=1)
c       call wavelet_forming(wavelet,lt,0.002,20.0,shift)
c            k=1
c              do ix = 1,lt
c               write(222,rec = k)wavelet(ix)
c                 write(*,*)wavelet(ix)
c                k = k+1
c                end do
c          do ix=1,lt
c          wavelet_dri(ix)=(wavelet(ix+2)+wavelet(ix)-2.0*
c     + wavelet(ix+1))*500*500
c          end do
c          k = 1
c          open(7777,file='./wavelet_cor.dat',access = 'direct',recl = 1)
c        call  covariance(wavelet_dri,lt,wavelet_cor)
c               do ix=1,lt
c                write(7777,rec = k)wavelet_cor(ix)
c                k = k + 1
c                end do

*========== ========= 2D Time to 3D Time ================         
c      open(777,file='time_temp.dat',access='direct',recl=1)
     
c      k = 1
      
c      do ix = 1, NZ*NX*NSHOT
c        read(777, rec=k)time2d(ix)
c        k = k + 1
c      enddo

c      do ix=0, NZ*NX*NSHOT-1
c           ishot_1=(ix/(NZ*NX))+1
c           ixx=(mod(ix,NZ*NX)/NZ)+1
c           izz=mod(ix,NZ)+1
c           time_s(ishot_1,ixx,izz)=time2d(ix+1)
c           time_r(ishot_1,ixx,izz)=time2d(ix+1)
C           write(*,*)"check time",ixx,izz,time_s(1,22,103)
c      enddo

       
c         CALL Hessian_construct(NSHOT, NZ, NX, time_s, time_r,
c     +                         wavelet_cor)

      RETURN
      
      END
      

*=============================================================================

      SUBROUTINE TRAVELTIME_2D(SS1, SS2, TT1, TT2, 
     +           SLOWNESS, TIME, NX, DX, NY, DY, 
     +           NS_X, NS_Y, DS_X, DS_Y)
 
      INTEGER NX, NY, NS_X, NS_Y
      REAL    DX, DY, DS_X, DS_Y
       
      DIMENSION SS1(0:NX), SS2(0:NX), TT1(NX), TT2(NX)
      DIMENSION SLOWNESS(0:NX, 0:NY), TIME(NX, NY)

*===== calculating the NS_Y layer travel time

      CALL START_LAYER(TIME, NX, DX, NY, DY,
     +       NS_X, NS_Y, SS1, TT1, SLOWNESS)

*===== calculating the start time value

      CALL UPWARD_CALCUL(TIME, NX, DX, NY, DY, 
     +            NS_X, NS_Y, DS_X, DS_Y, SS1,
     +            SS2, TT1, TT2, SLOWNESS)       

*===== Forward calculating the minimum traveltime  

      CALL FORWARD_CALCUL(TIME, NX, DX, NY, DY,  
     +             NS_X, NS_Y, DS_X, DS_Y, SS1, 
     +             SS2, TT1, TT2, SLOWNESS) 
     
*===== Backward calculating for replacing the forward calculated 
*===== minimum traveltime

      CALL BACKWARD_CALCUL(TIME, NX, DX, NY, DY, 
     +              DS_X, DS_Y, SS1, SS2, TT1, 
     +              TT2, SLOWNESS)

*===== Write the calculated minimum traveltime onto the diskfile

      RETURN
      END

*=================================================================

      SUBROUTINE START_LAYER(TIME, NX, DX, NY, DY,
     +             NS_X, NS_Y, SS1, TT1, SLOWNESS)
      
      REAL      DX, DY
      INTEGER   NX, NY, NS_X, NS_Y
      DIMENSION SS1(0:NX), TT1(NX)
      DIMENSION TIME(NX, NY), SLOWNESS(0:NX, 0:NY)
      
*===== calculating the start time value

      DO IX=0, NX
         SS1(IX)=SLOWNESS(IX, NS_Y-1)
c      write(*,*)'SS1,ix',IX,SS1(IX)
      END DO

      TEMP_T=0.0
      DO IX=NS_X-1, 1, -1
         TEMP_T=TEMP_T+DX*SS1(IX)
         TT1(IX)=TEMP_T
c         write(*,*)'TT1,ns_x',NS_X,TT1(IX)
      END DO

      TT1(NS_X)=0.0

      TEMP_T=0.0
      DO IX=NS_X+1, NX
         TEMP_T=TEMP_T+DX*SS1(IX-1)
         TT1(IX)=TEMP_T
c         write(*,*)'TT111',TT1(IX)
      END DO

      DO IX=1, NX
         TIME(IX, NS_Y)=TT1(IX)
         write(*,*)'ix,ns_y,TT1',IX,NS_Y,TT1(IX)
      END DO

      RETURN
      END

*=================================================================

      SUBROUTINE UPWARD_CALCUL(TIME, NX, DX, NY, DY,
     +           NS_X, NS_Y, DS_X, DS_Y, SS1, SS2, 
     +           TT1, TT2, SLOWNESS)

      REAL    T1, T2, X1, X2, Y1, Y2
      REAL    DX, DY, DS_X, DS_Y
      INTEGER NX, NY, NS_X, NS_Y
      DIMENSION  SS1(0:NX), SS2(0:NX), TT1(NX), TT2(NX)
      DIMENSION TIME(NX, NY), SLOWNESS(0:NX, 0:NY)

      
*=================================================
      DO IX=1, NX
         TT1(IX)=TIME(IX, NS_Y)
c         write(*,*)'ns_y,TT1',NS_Y,TT1(IX)
      END DO

      DO 7777 IY=NS_Y-1 ,1, -1
c       print *,'IY',IY  
*======  get the slowness of each layer
      
       DO IX=0, NX
          SS1(IX)=SLOWNESS(IX, IY)
          SS2(IX)=SLOWNESS(IX, IY-1)
          write(*,*)'ix,iy,ss1,ss2',IX,IY,NS_Y
       END DO
C          write(*,*)'ix,iy,ss1,ss2',IX,IY,SS1(IX),SS2(IX)

c          write(*,*)'ix,iy,ss1,ss2',IX,IY
       DO IX=1, NX
          TT2(IX)=100000.0
       END DO

      IF(IY.EQ.(NS_Y-1))  THEN
          TEMP_T=AMIN1(SLOWNESS(NS_X-1, NS_Y-1), 
     +                 SLOWNESS(NS_X, NS_Y-1)) * DY
          TT2(NS_X)=TEMP_T
          TEMP_T=SQRT(DX*DX + DY*DY)*SLOWNESS(NS_X-1, NS_Y-1)
          TT2(NS_X-1)=TEMP_T
          TEMP_T=SQRT(DX*DX + DY*DY)*SLOWNESS(NS_X, NS_Y-1)
          TT2(NS_X+1)=TEMP_T
      END IF
   
*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        DO IX=2, NX
         
           T1=TT1(IX-1)
           T2=TT1(IX)
           X1=ABS((IX-1)*DX - DS_X)
           X2=ABS(IX*DX     - DS_X)

         IF(IX.EQ.NS_X) THEN
           TT2(NS_X)=T2+AMIN1(SS1(IX-1), SS1(IX))*DY
         END IF

             
         IF(SS1(IX).LT.SS1(IX-1)) THEN     

            TS=T2+SS1(IX)*DY

         ELSE
        
           SNESS=SS1(IX-1)
           
           W=(T2*T2-T1*T1)/(X2*X2-X1*X1)
           CALL ROOT1(X0, W, SNESS, X1, X2, T1, T2, DY)
           T0=SQRT(W*(X0*X0 - X1*X1) + T1*T1)
           TS=T0+SNESS*SQRT((X2-X0)*(X2-X0) + DY*DY)

        END IF

           IF(TS.LT.TT2(IX)) TT2(IX)=TS

        END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DO IX=NX-1, 1, -1
 
           T1=TT1(IX+1)
           T2=TT1(IX)
           X1=ABS((IX+1)*DX-DS_X)
           X2=ABS(IX*DX-DS_X)

         IF(SS1(IX-1).LT.SS1(IX)) THEN     

            TS=T2+SS1(IX-1)*DY

         ELSE
        
           SNESS=SS1(IX)
           
           W=(T2*T2-T1*T1)/(X2*X2-X1*X1)
           CALL ROOT1(X0, W, SNESS, X1, X2, T1, T2, DY)
           T0=SQRT(W*(X0*X0-X1*X1)+T1*T1)
           TS=T0+SNESS*SQRT((X2-X0)*(X2-X0)+DY*DY)

        END IF

           IF(TS.LT.TT2(IX)) TT2(IX)=TS

         END DO

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DO IX=2, NX
           
            T1=TT1(IX-1)
            T2=TT2(IX-1)
            Y1=ABS(IY*DY-DS_Y)
            Y2=ABS((IY-1)*DY-DS_Y)

        IF(SS2(IX-1).LT.SS1(IX-1)) THEN

           TS=T2+SS2(IX-1)*DX

        ELSE

            SNESS=SS1(IX-1)
            
            W=(T2*T2-T1*T1)/(Y2*Y2-Y1*Y1)
            CALL ROOT1(Y0, W, SNESS, Y1, Y2, T1, T2, DX)
            T0=SQRT(W*(Y0*Y0-Y1*Y1)+T1*T1)
            TS=T0+SNESS*SQRT((Y2-Y0)*(Y2-Y0)+DX*DX)

        END IF

            IF(TS.LT.TT2(IX)) TT2(IX)=TS
         
         END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

         DO IX=NX-1, 1, -1
          
          T1=TT1(IX+1)
          T2=TT2(IX+1)
          Y1=ABS(IY*DY-DS_Y)
          Y2=ABS((IY-1)*DY-DS_Y)

        IF(SS2(IX).LT.SS1(IX)) THEN

          TS=T2+SS2(IX)*DX

        ELSE
        
          SNESS=SS1(IX)
          
          W=(T2*T2-T1*T1)/(Y2*Y2-Y1*Y1)
          CALL ROOT1(Y0, W, SNESS, Y1, Y2, T1, T2, DX)
          T0=SQRT(W*(Y0*Y0-Y1*Y1)+T1*T1)
          TS=T0+SNESS*SQRT((Y2-Y0)*(Y2-Y0)+DX*DX)

        END IF

        IF(TS.LT.TT2(IX)) TT2(IX)=TS

        END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
          DO IX=1, NX
             TT1(IX)=TT2(IX)
        END DO

          DO IX=1, NX
             TIME(IX, IY)=TT2(IX)
        END DO

7777  CONTINUE

        RETURN
        END


*====================================================================

      SUBROUTINE FORWARD_CALCUL(TIME, NX, DX, NY, DY,  
     +           NS_X, NS_Y, DS_X, DS_Y, SS1, SS2,
     +           TT1, TT2, SLOWNESS)

      REAL      T1, T2, X1, X2, Y1, Y2
      REAL      DX, DY, DS_X, DS_Y
      INTEGER   NX, NY, NS_X, NS_Y
      DIMENSION SS1(0:NX), SS2(0:NX), TT1(NX), TT2(NX)
      DIMENSION TIME(NX, NY), SLOWNESS(0:NX, 0:NY)

*============================================================= 

      DO IX=1,NX
         TT1(IX)=TIME(IX,1)
      END DO

      DO 8888 IY=2, NY
        
*====== GET THE SLOWNESS OF EACH LAYER
      
       DO IX=0, NX
         SS1(IX)=SLOWNESS(IX, IY-1)
         SS2(IX)=SLOWNESS(IX, IY)
       END DO

*===== assigning the (beyond) maximum traveltime for the
*===== calculating traveltime

      IF(IY.LE.NS_Y) THEN

         DO IX=1, NX
            TT2(IX)=TIME(IX, IY)
         END DO     

      ELSE

         DO IX=1, NX
            TT2(IX)=1000000.0
         END DO

      ENDIF

*====== calculating the travel time around the source
      
      IF(IY.EQ.(NS_Y+1)) THEN

        TEMP_T=DY*AMIN1(SLOWNESS(NS_X, NS_Y), 
     +                  SLOWNESS(NS_X-1, NS_Y))
        TT2(NS_X)=TEMP_T

        TEMP_T=SQRT(DX*DX+DY*DY)*SLOWNESS(NS_X-1, NS_Y)
        TT2(NS_X+1)=TEMP_T

        TEMP_T=SQRT(DX*DX+DY*DY)*SLOWNESS(NS_X, NS_Y) 
        TT2(NS_X-1)=TEMP_T

      END IF

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DO IX=2, NX

            T1=TT1(IX-1)
            T2=TT1(IX)
            X1=ABS((IX-1)*DX-DS_X)
            X2=ABS((IX)*DX-DS_X)

         IF(IY.GT.NS_Y+1.AND.IX.EQ.NS_X) THEN
            TT2(NS_X)=T2+AMIN1(SS1(IX-1), SS1(IX))*DY
         ENDIF

         IF(SS1(IX).LT.SS1(IX-1)) THEN     

            TS=T2+SS1(IX)*DY

         ELSE

            SNESS=SS1(IX-1)
          
            W=(T2*T2-T1*T1)/(X2*X2-X1*X1)
            CALL ROOT1(X0, W, SNESS, X1, X2, T1, T2, DY)
            T0=SQRT(W*(X0*X0-X1*X1)+T1*T1)
            TS=T0+SNESS*SQRT((X2-X0)*(X2-X0)+DY*DY)

        END IF

           IF(TS.LT.TT2(IX)) TT2(IX)=TS

        END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DO IX=NX-1, 1, -1

          T1=TT1(IX+1)
          T2=TT1(IX)
          X1=ABS((IX+1)*DX-DS_X)
          X2=ABS(IX*DX-DS_X)

         IF(SS1(IX-1).LT.SS1(IX)) THEN     

            TS=T2+SS1(IX-1)*DY

         ELSE
       
          SNESS=SS1(IX)
          
          W=(T2*T2-T1*T1)/(X2*X2-X1*X1)
          CALL ROOT1(X0, W, SNESS, X1, X2, T1, T2, DY)
          T0=SQRT(W*(X0*X0-X1*X1)+T1*T1)
          TS=T0+SNESS*SQRT((X2-X0)*(X2-X0)+DY*DY)

        END IF

          IF(TS.LT.TT2(IX)) TT2(IX)=TS

         END DO
         
*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DO IX=2, NX

           T1=TT1(IX-1)
           T2=TT2(IX-1)
           Y1=ABS((IY-1)*DY-DS_Y)
           Y2=ABS((IY)*DY-DS_Y)

        IF(SS2(IX-1).LT.SS1(IX-1)) THEN

           TS=T2+SS2(IX-1)*DX

        ELSE

           SNESS=SS1(IX-1)
           
           W=(T2*T2-T1*T1)/(Y2*Y2-Y1*Y1)
           CALL ROOT1(Y0, W, SNESS, Y1, Y2, T1, T2, DX)
           T0=SQRT(W*(Y0*Y0-Y1*Y1)+T1*T1)
           TS=T0+SNESS*SQRT((Y2-Y0)*(Y2-Y0)+DX*DX)

        END IF

           IF(TS.LT.TT2(IX)) TT2(IX)=TS

         END DO
          
*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

         DO IX=NX-1, 1, -1

            T1=TT1(IX+1)
            T2=TT2(IX+1)
            Y1=ABS((IY-1)*DY-DS_Y)
            Y2=ABS((IY)*DY-DS_Y)

          IF(SS2(IX).LT.SS1(IX)) THEN

            TS=T2+SS2(IX)*DX

          ELSE

            SNESS=SS1(IX)
           
            W=(T2*T2-T1*T1)/(Y2*Y2-Y1*Y1)
            CALL ROOT1(Y0, W, SNESS, Y1, Y2, T1, T2, DX)
            T0=SQRT(W*(Y0*Y0-Y1*Y1)+T1*T1)
            TS=T0+SNESS*SQRT((Y2-Y0)*(Y2-Y0)+DX*DX)

         END IF

            IF(TS.LT.TT2(IX)) TT2(IX)=TS

          END DO
          
*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           
          DO IX=1, NX
             TT1(IX)=TT2(IX)
          END DO

          DO IX=1, NX
             TIME(IX, IY)=TT2(IX)
          END DO

8888   CONTINUE
         
        
        RETURN
        END

*===========================================================================
    
        SUBROUTINE BACKWARD_CALCUL(TIME, NX, DX, NY, DY, 
     +       DS_X, DS_Y, SS1, SS2, TT1, TT2, SLOWNESS)

        REAL    T1, T2, X1, X2, Y1, Y2
        REAL    DX, DY, DS_X, DS_Y
        INTEGER NX, NY
        DIMENSION SS1(0:NX), SS2(0:NX), TT1(NX), TT2(NX)
        DIMENSION TIME(NX, NY), SLOWNESS(0:NX, 0:NY)
        
*==================================================

      DO IX=1, NX
        TT1(IX)=TIME(IX, NY)
      END DO

      DO 9999 IY=NY-1, 1, -1

*======  get the slowness of each layer
      
       DO IX=0, NX
          SS1(IX)=SLOWNESS(IX, IY)
          SS2(IX)=SLOWNESS(IX, IY-1)
       END DO

       DO IX=1, NX
          TT2(IX)=TIME(IX, IY)
       END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        DO IX=2, NX
         
           T1=TT1(IX-1)
         T2=TT1(IX)
         X1=ABS((IX-1)*DX-DS_X)
         X2=ABS((IX)*DX-DS_X)

         IF(SS1(IX).LT.SS1(IX-1)) THEN     

            TS=T2+SS1(IX)*DY

         ELSE
        
           SNESS=SS1(IX-1)
           
           W=(T2*T2-T1*T1)/(X2*X2-X1*X1)
           CALL ROOT1(X0, W, SNESS, X1, X2, T1, T2, DY)
           T0=SQRT(W*(X0*X0-X1*X1)+T1*T1)
           TS=T0+SNESS*SQRT((X2-X0)*(X2-X0)+DY*DY)

        END IF

           IF(TS.LT.TT2(IX)) TT2(IX)=TS

        END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DO IX=NX-1, 1, -1
 
           T1=TT1(IX+1)
           T2=TT1(IX)
           X1=ABS((IX+1)*DX-DS_X)
           X2=ABS((IX)*DX-DS_X)

         IF(SS1(IX-1).LT.SS1(IX)) THEN     

            TS=T2+SS1(IX-1)*DY

         ELSE
        
           SNESS=SS1(IX)
           
           W=(T2*T2-T1*T1)/(X2*X2-X1*X1)
           CALL ROOT1(X0, W, SNESS, X1, X2, T1, T2, DY)
           T0=SQRT(W*(X0*X0-X1*X1)+T1*T1)
           TS=T0+SNESS*SQRT((X2-X0)*(X2-X0)+DY*DY)

        END IF

           IF(TS.LT.TT2(IX)) TT2(IX)=TS

         END DO

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DO IX=2, NX
           
            T1=TT1(IX-1)
          T2=TT2(IX-1)
          Y1=ABS(IY*DY-DS_Y)
          Y2=ABS((IY-1)*DY-DS_Y)

        IF(SS2(IX-1).LT.SS1(IX-1)) THEN

           TS=T2+SS2(IX-1)*DX

        ELSE

            SNESS=SS1(IX-1)
            
            W=(T2*T2-T1*T1)/(Y2*Y2-Y1*Y1)
            CALL ROOT1(Y0, W, SNESS, Y1, Y2, T1, T2, DX)
            T0=SQRT(W*(Y0*Y0-Y1*Y1)+T1*T1)
            TS=T0+SNESS*SQRT((Y2-Y0)*(Y2-Y0)+DX*DX)

        END IF

            IF(TS.LT.TT2(IX)) TT2(IX)=TS
         
         END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

         DO IX=NX-1, 1, -1
          
          T1=TT1(IX+1)
        T2=TT2(IX+1)
        Y1=ABS(IY*DY-DS_Y)
        Y2=ABS((IY-1)*DY-DS_Y)

        IF(SS2(IX).LT.SS1(IX)) THEN

          TS=T2+SS2(IX)*DX

        ELSE
        
          SNESS=SS1(IX)
          
          W=(T2*T2-T1*T1)/(Y2*Y2-Y1*Y1)
          CALL ROOT1(Y0, W, SNESS, Y1, Y2, T1, T2, DX)
          T0=SQRT(W*(Y0*Y0-Y1*Y1)+T1*T1)
          TS=T0+SNESS*SQRT((Y2-Y0)*(Y2-Y0)+DX*DX)

        END IF
      
          IF(TS.LT.TT2(IX)) TT2(IX)=TS
         
        END DO

*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
          DO IX=1, NX
             TT1(IX)=TT2(IX)
        END DO

          DO IX=1, NX
             TIME(IX, IY)=TT2(IX)
        END DO

9999  CONTINUE

        RETURN
        END

*=============================================================================
           
      SUBROUTINE ROOT1(RTBIS, W, S, X1, X2, T1, T2, DY)

      REAL W, S, T1, T2, DY

      INTEGER JMAX
      REAL rtbis,x1,x2,xacc, func
      EXTERNAL func
      PARAMETER(xacc=1.0E-8)
      PARAMETER (JMAX=40)
      INTEGER j
      REAL dx,f,fmid,xmid

      fmid=func(x2, W, S, X1, X2, T1, DY)
      f=func(x1, W,  S, X1, X2, T1, DY)

      if(f*fmid.ge.0.) then
    
        rtbis=x1
      return

      end if

      if(f.lt.0.)then
        rtbis=x1
        dx=x2-x1
      else
        rtbis=x2
        dx=x1-x2
      end if 

      do 11 j=1,JMAX

        dx=dx*.5
        xmid=rtbis+dx

        fmid=func(xmid, W, S, X1, X2, T1, DY)

        if(fmid.le.0.) rtbis=xmid

        if(abs(dx).lt.xacc .or. fmid.eq.0.) then
        return
        end if

11    continue

      pause 'too many bisections in rtbis'
      END

*==========================================================================

      REAL FUNCTION FUNC(X, W, S, X1, X2, T1, DY)

      REAL X
      REAL S, X1, X2, T1, DY
      REAL W
      REAL A, B, C, D, E
       
c      write(*,*) 'x=', x

      A=W-W*W/S/S
      B=-2.0*X2*A
      C=W*X2*X2-W*X1*X1+T1*T1-(W*W/S/S)*(X2*X2+DY*DY)
      D=2.0*X2*(W*X1*X1-T1*T1)
      E=X2*X2*(T1*T1-W*X1*X1)

      FUNC=A*X*X*X*X+B*X*X*X+C*X*X+D*X+E

      END




*============================================================================
             
      SUBROUTINE ZERO_BUF(NX, NZ, SS1, SS2, TT1, TT2, TIME, SLOWNESS)

      INTEGER NX, NZ
      DIMENSION SS1(0:NX), SS2(0:NX), TT1(NX), TT2(NX)
      DIMENSION TIME(NX, NZ), SLOWNESS(0:NX, 0:NZ)

*=======================================================================
      
c      write(*,*)  'NX, NZ=', NX, NZ
      
*=======================================================================      
      DO IX=0, NX
         SS1(IX)=0.0
         SS2(IX)=0.0
      END DO

      DO IX=1, NX
         TT1(IX)=0.0
         TT2(IX)=0.0
      END DO

      DO IX=1, NX
         DO IZ=1, NZ
            TIME(IX, IZ)=100000.0
         END DO
      END DO

      DO IX=0, NX
         DO IZ=0, NZ
            SLOWNESS(IX, IZ)=0.0
         END DO
      END DO

      RETURN
      END

*=================================================================

      SUBROUTINE READ_CURRENT_SHOT_VELO(NS_X, NXS_left, NXS_right,
     +                  NVX, NX, NVZ, NZ, SLOWNESS)

      INTEGER NX, NZ, NS_X, NXS_left, NXS_right, K
      DIMENSION SLOWNESS(0:NX, 0:NZ),VEL(0:NX, 0:NZ)

      REAL      HEAD(60)

*==============================================================

c      write(*,*) 'NS_X, NXS_left, NXS_right, NX, NZ,='
c      write(*,*)  NS_X, NXS_left, NXS_right,NX, NZ

*===== read the velocity field into the working buffer SLOWNESS
      K = 1
      DO IX=0, NX
       DO IZ=0, NZ
        SLOWNESS(IX, IZ)=10.0  ! assignning a small value avioding
                                   ! error velocity inputted
       END DO
      END DO
      
      IIX=1      
      DO 66 IX=NS_X-NXS_left, NS_X+NXS_right
c      READ(11, REC=IX) (HEAD(Ih), Ih=1, 60),
c    +                  (SLOWNESS(IIX, IZ), IZ=1, NZ)

       IXX=IX
       IF(IXX.LT.1) IXX=1
       IF(IXX.GT.NVX) IXX=NVX

c      write(*,*) 'IXX=', IXX
c       READ(11, REC=IXX)(HEAD(Ih),Ih=1,60),(SLOWNESS(IIX, IZ), 
c     +                  IZ=1, NZ)
       IIX=IIX+1       
66    CONTINUE
      open(88888,file = './data/marmousi2_z581x1502_sm.dat',access=
     +'direct',recl=1)
      DO IX=1, NX
       DO IZ=1, NZ
c        SLOWNESS(IX, IZ)=1.0/2500.0
        read(88888,rec = K)VEL(IX,IZ)
        SLOWNESS(IX,IZ) = 1/VEL(IX, IZ)
        K = K+1
        IF(IZ.EQ.1)SLOWNESS(IX, IZ-1) = SLOWNESS(IX, IZ)
        IF(IX.EQ.1)SLOWNESS(IX-1, IZ) = SLOWNESS(IX, IZ)
       END DO
      END DO

      RETURN
      END

*==========================================================================
*======  write the calculated travel time onto the disk file FN2

        SUBROUTINE WRITE_DISK(NX, NZ, Krec_save, TIME)

        INTEGER NX, NZ, Krec_save
        DIMENSION TIME(NX, NZ)
        CHARACTER*256 FN2

c        DO IX=1, NX
c           DO IZ=1, NZ
c             WRITE(13, *) IX, IZ, TIME(IX, IZ)
c           END DO
c        END DO

       do ix=1, nx
          write(12, rec=Krec_save) (time(ix, iz), iz=1, nz)
          Krec_save = Krec_save + 1
       end do

c       open(45, file='time_shot')
c       do ix=1, nx
c        do iz=1, nz       
c        write(45,*) ix, iz, time(ix, iz)
c        end do        
c       end do       
c       close(45)

       RETURN
       END

*===========================================================================
*====== construct the Hessian Matrix save into the disk file
c         SUBROUTINE Hessian_construct(NSHOT,NZ,NX,time_s,time_r,
c     +                           wavelet_cor,lt )

c        INTEGER ipsfx, jpsfz, ishot, itrace, ix, jz, k, ii, jj
c        real ttt, tt, aa, rtime_s, rtime_r,tt_m
c        dimension time_s(NSHOT,NX,NZ),time_r(NSHOT,NX,NZ)
c        dimension H(NX,NZ),wavelet_cor(lt)
C        allocate(time_s(NSHOT,NX,NZ))
c        allocate(time_r(NSHOT,NX,NZ))
c        H = 0.0 
c        k = 1
c        write(*,*)"times timer",NSHOT
c        do ishot=1,NSHOT,1
c        do itrace=1,NSHOT,1

c        do ix=31,NX-10,21
c        do ipsfx= ix-10,ix+10
c        do jz=11,NZ-10,21

c        write(*,*)"check",ipsfx
c        write(*,*)"times timer",time_s(10,1,1),time_r(10,1,1)
c        do jz=31,NZ-10,21
c        write(*,*)"check",jz,ipsfx,ix
c        write(*,*)"times timer",time_s(10,1,1),time_r(10,1,1)
c        do jpsfz = jz-10,jz+10
c             if(time_s(ishot,ipsfx,jpsfz)==0.or.
c     +  time_r(itrace,ipsfx,jpsfz)==0.or.time_s(ishot,ix,jz)==0
c     + .or.  time_r(itrace,ix,jz)==0)cycle
c        write(*,*)"times timer",jz,ix,time_s(10,1,1),time_r(10,1,1)

c               write(*,*)"check",time_s(ishot,ipsfx,jpsfz),ipsfx,jpsfz
c               tt=time_s(ishot,ipsfx,jpsfz) + time_r(itrace,ipsfx,jpsfz)
c               write(*,*)"check",jz,ix
c               tt_m=time_s(ishot,ix,jz)+time_r(itrace,ix,jz)
c               rtime_s=1./time_s(itrace,ipsfx,jpsfz)
c               rtime_r=1./time_r(itrace,ipsfx,jpsfz)
c               ttt=abs((tt-tt_m))*500
c              if(ttt.lt.1)then
c                       ttt=ttt+1
c              endif
c               aa=tt*(rtime_s*rtime_s+rtime_r*rtime_r)
c          aa_2=1./(time_s(ishot_1,ix,jz)*time_r(itrace,ix,jz))

c               write(*,*)'tt,tt_m,ttt,ix,jz,ipsfx,jpsfz'
c               write(*,*),tt,tt_m,ttt,ix,jz,ipsfx,jpsfz,NZ,NX
c               H(ipsfx,jpsfz)=H(ipsfx,jpsfz)+aa*wavelet_cor(ttt)
c               write(*,*)"check H",H(ipsfx,jpsfz),wavelet_cor(ttt)
c        end do
c        end do
c        end do
c        end do
c        end do
c             write(*,*)"iiishot is  done",ishot,tt
c        end do
c        open(888,file='Hessian1.dat',access='direct',recl=1)
c        do ii=1,NX,1
c           do jj=1,NZ,1
c            write(888,rec=k)H(ii,jj)
c           k = k+1
c         end do
c         end do
         




c       RETURN
c       END

